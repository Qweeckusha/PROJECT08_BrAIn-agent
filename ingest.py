import json
import uuid
import os
from datetime import datetime
from typing import List

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# ==========================================
#                   КОНФИГ
# ==========================================
LM_STUDIO_URL = "http://localhost:1234/v1"
GENERATION_MODEL = "qwen/qwen3-4b-2507"  # Проверь название в LM Studio
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BRAIN_FILE = "brain.json"
FAISS_INDEX = "faiss_index"


# ==========================================
# PYDANTIC СХЕМА (для парсинга JSON)
# ==========================================
class KnowledgeEntry(BaseModel):
    """Структура для извлечённого знания"""
    text: str = Field(description="Основная мысль, можно структурировать, но не удалять важные детали")
    topic: str = Field(
        description="Общая тема, например: ML.classification.RandomForest (макс. глубина 2, через точку)")
    subtopic: str = Field(description="Уточнение темы, можно микс русского и английского")
    tags: List[str] = Field(description="Список тегов")
    level: str = Field(description="Уровень сложности: beginner|intermediate|advanced")
    status: str = Field(default="draft", description="Статус записи")


# ==========================================
# 🧠 ИНИЦИАЛИЗАЦИЯ КОМПОНЕНтов LANGCHAIN
# ==========================================
print("📦 Загрузка LangChain компонентов...")

# 1. Эмбеддер (тот же, что и раньше, но через обёртку LC)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 2. LLM (через OpenAI-совместимый интерфейс для LM Studio)
llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio",  # Фиктивный ключ
    model=GENERATION_MODEL,
    temperature=0.5,
    max_tokens=1024
)

# 3. Парсер JSON (автоматически добавит format_instructions в промпт)
json_parser = JsonOutputParser(pydantic_object=KnowledgeEntry)

# ==========================================
# 📝 ПРОМПТЫ
# ==========================================

# Промпт для парсинга (извлечение структуры из текста)
PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — парсер знаний. Извлеки информацию из сообщения пользователя.
Верни ТОЛЬКО валидный JSON без markdown и пояснений. Если информации нет, верни пустой объект.

{format_instructions}

Важно:
- topic: краткая иерархия через точку, макс. 2 уровня (ML.classification)
- subtopic: уточнение на русском/английском
- tags: 3-5 релевантных тегов
- level: один из [beginner, intermediate, advanced]"""),
    ("human", "{input}"),
])

# Промпт для валидации (дубликат/дополнение/новое)
VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — редактор базы знаний. Сравни НОВЫЙ текст с СУЩЕСТВУЮЩИМИ записями.

Если НОВЫЙ текст просто перефразирует существующий (смысл тот же) -> 'duplicate'
Если НОВЫЙ текст добавляет детали, уточнения или факты, которых нет в старых -> 'complement'  
Если НОВЫЙ текст о совершенно другом -> 'new'

Ответь ТОЛЬКО одним словом: duplicate, complement или new."""),
    ("human", """СУЩЕСТВУЮЩИЕ ЗАПИСИ:
{context}

НОВЫЙ ТЕКСТ:
"{new_text}"

Решение:"""),
])

# ==========================================
# 🔗 ЦЕПОЧКИ (CHAINS)
# ==========================================

# Цепочка парсинга: промпт → LLM → JSON-парсер
parse_chain = PARSE_PROMPT | llm | json_parser

# Цепочка валидации: промпт → LLM → строковый парсер
validate_chain = VALIDATE_PROMPT | llm | StrOutputParser()


# ==========================================
# 🗄️ РАБОТА С ДАННЫМИ (Raw Python для кастомной логики)
# ==========================================

def load_brain() -> list:
    """Загружает brain.json или возвращает пустой список"""
    try:
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_brain(entries: list):
    """Сохраняет brain.json"""
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def load_or_create_faiss() -> FAISS:
    """Загружает индекс FAISS или создаёт новый"""
    if os.path.exists(FAISS_INDEX):
        return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        # Создаём пустой индекс
        return FAISS.from_texts([], embeddings)


def find_similar_lc(query_text: str, db: FAISS, threshold: float = 0.60, top_k: int = 10) -> list:
    """
    LangChain-версия поиска похожих.
    Возвращает список записей с сходством > threshold.
    """
    # similarity_search_with_score возвращает (Document, score)
    # Для all-MiniLM + FAISS с нормализованными векторами: выше = лучше (inner product ≈ косинус)
    docs_with_scores = db.similarity_search_with_score(query_text, k=top_k)

    similar = []
    for doc, score in docs_with_scores:
        if score > threshold:  # higher = more similar
            similar.append({
                "id": doc.metadata.get("id"),
                "similarity": float(score),
                "text": doc.page_content
            })

    return sorted(similar, key=lambda x: x["similarity"], reverse=True)


def decide_with_llm_lc(new_text: str, existing_entries: list) -> str:
    """LangChain-версия валидации через LLM"""
    context = "\n".join([f"- {e['text']}" for e in existing_entries])

    try:
        response = validate_chain.invoke({
            "context": context,
            "new_text": new_text
        })
        decision = response.strip().lower()
        return decision if decision in ["duplicate", "complement", "new"] else "new"
    except Exception as e:
        print(f"⚠️ Ошибка валидации: {e}")
        return "new"  # При ошибке — сохраняем, лучше продублировать


def save_knowledge_lc(parsed: dict, db: FAISS) -> str:
    """
    Сохраняет запись в brain.json и обновляет FAISS индекс.
    Возвращает ID новой записи.
    """
    entry_id = str(uuid.uuid4())
    related = parsed.get("related_ids", [])

    # Формируем запись для brain.json (твоя кастомная схема)
    entry = {
        "id": entry_id,
        "text": parsed["text"],
        "topic": parsed["topic"],
        "subtopic": parsed["subtopic"],
        "tags": parsed["tags"],
        "level": parsed.get("level", "intermediate"),
        "status": "draft",
        "created": datetime.now().isoformat(),
        "related_ids": related
    }

    # 1. Сохраняем в brain.json
    brain = load_brain()
    brain.append(entry)
    save_brain(brain)

    # 2. Добавляем в векторный индекс (FAISS)
    # add_texts возвращает список ID, но мы передаём свой через ids=
    db.add_texts(
        texts=[parsed["text"]],
        metadatas=[{"id": entry_id, "topic": parsed["topic"]}],
        ids=[entry_id]
    )
    db.save_local(FAISS_INDEX)

    print(f"✅ Сохранено: {entry['text'][:40]}...")
    return entry_id


# ==========================================
# 🚀 ГЛАВНЫЙ ЦИКЛ
# ==========================================

def main():
    print("🧠 BrAIn Ingest (LangChain Edition)")
    print("Вводи текст (или 'q' для выхода)")
    print("-" * 30)

    # Загружаем или создаём векторный индекс
    db = load_or_create_faiss()

    while True:
        user_input = input("\n📝 Ты: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue

        # 1. Парсинг через LLM
        print("⏳ Парсю смысл...")
        try:
            # format_instructions автоматически добавит JSON-схему в промпт
            parsed = parse_chain.invoke({
                "input": user_input,
                "format_instructions": json_parser.get_format_instructions()
            })

            # Валидация результата парсинга
            if not parsed or not all(k in parsed for k in ["text", "topic", "tags"]):
                print("❌ Не удалось распарсить JSON. Попробуй чётче.")
                print(f"Debug: {parsed}")
                continue

        except Exception as e:
            print(f"❌ Ошибка парсинга: {e}")
            # Fallback: пробуем распарсить вручную, если модель вернула "грязный" JSON
            continue

        # 2. Векторизация (через LangChain эмбеддер)
        print("🔢 Векторизую...")
        # embed_query возвращает list[float] — то, что нужно для FAISS

        # 3. Поиск похожих (порог ниже, чтобы ловить связи)
        print("🔍 Проверяю на связи...")
        similar = find_similar_lc(parsed["text"], db, threshold=0.60)

        if not similar:
            print("✨ Новое: Похожих тем не найдено.")
            decision = "new"
        else:
            best_match = similar[0]
            score = best_match['similarity']
            print(f"🔗 Найдено: {best_match['text'][:50]}... (сходство: {score:.2f})")

            # Логика зон (твоя бизнес-логика остаётся raw)
            if score > 0.85:
                print("⚠️ Высокое сходство. Проверяю на дубликат...")
                decision = decide_with_llm_lc(parsed["text"], similar)
            elif score > 0.60:
                print("💡 Интересная связь. Это дополнение к существующей теме.")
                decision = "complement"
                # Автоматически линкуем
                if "related_ids" not in parsed:
                    parsed["related_ids"] = []
                parsed["related_ids"].append(best_match["id"])
            else:
                decision = "new"

        # 4. Действие по решению
        if decision == "duplicate":
            print("♻️ Это дубликат. Пропускаем сохранение.")
            continue
        elif decision == "complement":
            print(f"🔗 Сохраняю как дополнение к {best_match['id'][:8]}")
        else:
            print("✅ Сохраняю как новую запись.")

        # 5. Финальное сохранение
        save_knowledge_lc(parsed, db)


if __name__ == "__main__":
    main()
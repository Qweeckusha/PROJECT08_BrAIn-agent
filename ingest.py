import json
import uuid
import os
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
#               КОНФИГУРАЦИЯ
# ==========================================
LM_STUDIO_URL = "http://localhost:1234/v1"
GENERATION_MODEL = "qwen/qwen3-4b-2507"  # ← Проверь название в LM Studio
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BRAIN_FILE = "brain.json"
FAISS_INDEX = "faiss_index"

# ==========================================
#         ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ==========================================
print("📦 Загрузка LangChain компонентов...")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

llm = ChatOpenAI(
    base_url=LM_STUDIO_URL,
    api_key="lm-studio",
    model=GENERATION_MODEL,
    temperature=0.5,
    max_tokens=1024
)

output_parser = StrOutputParser()

# ==========================================
#                  ПРОМПТЫ
# ==========================================

PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — парсер знаний. Извлеки информацию из сообщения пользователя.
Верни ТОЛЬКО валидный JSON без markdown и пояснений. Если информации нет, верни пустой объект {{}}.

Требуемая структура JSON:
{{
  "text": "основная мысль, можно структурировать, но не удалять важные детали",
  "topic": "общая тема, например: ML.classification.RandomForest (макс. 2 уровня, через точку)",
  "subtopic": "уточнение темы, можно микс русского и английского",
  "tags": ["тег1", "тег2", "тег3"],
  "level": "beginner|intermediate|advanced",
  "status": "draft"
}}"""),
    ("human", "{input}"),
])

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

Решение:""")
])


# ==========================================
#                  CHAINS
# ==========================================

parse_chain = PARSE_PROMPT | llm | output_parser
validate_chain = VALIDATE_PROMPT | llm | output_parser


# ==========================================
#         ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def load_brain() -> list:
    """Загружает brain.json или возвращает пустой список"""
    try:
        with open(BRAIN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_brain(entries: list):
    """Сохраняет в brain.json"""
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def load_or_create_faiss() -> FAISS:
    """Загружает индекс FAISS или создаёт новый"""
    if os.path.exists(FAISS_INDEX):
        return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        # Создаём пустой индекс
        return FAISS.from_texts([], embeddings)


def find_similar_lc(query_text: str, db: FAISS, threshold: float = 0.60, top_k: int = 5) -> list:
    """
    LangChain-версия поиска похожих.
    Возвращает список записей с сходством > threshold.
    """
    # similarity_search_with_score возвращает (Document, score)
    docs_with_scores = db.similarity_search_with_score(query_text, k=top_k)

    similar = []
    for doc, score in docs_with_scores:
        # Для all-MiniLM + FAISS с нормализованными векторами: выше = лучше
        if score > threshold:
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

    # Схема записи в brain
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

    # Сохраняем в brain.json
    brain = load_brain()
    brain.append(entry)
    save_brain(brain)

    # Добавляем в векторный индекс (FAISS)
    db.add_texts(
        texts=[parsed["text"]],
        metadatas=[{"id": entry_id, "topic": parsed["topic"]}],
        ids=[entry_id]
    )
    db.save_local(FAISS_INDEX)

    print(f"✅ Сохранено: {entry['text'][:40]}...")
    return entry_id


# ==========================================
#                   main
# ==========================================

def main():
    print("🧠 BrAIn Ingest (LangChain Edition)")
    print("Вводи текст (или 'q' для выхода)")
    print("-" * 30)

    # Загружаем векторы
    db = load_or_create_faiss()

    while True:
        user_input = input("\nuser: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue

        print("⏳ Парсю смысл...")
        try:
            raw_response = parse_chain.invoke({"input": user_input})
            clean = raw_response.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            # Валидация результата парсинга
            if not parsed or not all(k in parsed for k in ["text", "topic", "tags"]):
                print("❌ Не удалось распарсить JSON. Попробуй чётче.")
                print(f"Debug: {parsed}")
                continue

        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Raw ответ модели: {raw_response[:200]}...")
            continue
        except Exception as e:
            print(f"❌ Ошибка парсинга: {e}")
            continue

        print("Векторизую и проверяю связи...")

        similar = find_similar_lc(parsed["text"], db, threshold=0.60)

        if not similar:
            print("NEW: Похожих тем не найдено.")
            decision = "new"
        else:
            best_match = similar[0]
            score = best_match['similarity']
            print(f"🔗 Найдено: {best_match['text'][:50]}... (сходство: {score:.2f})")

            # Логика зон
            if score > 0.85:
                print("Высокое сходство. Проверяю на дубликат...")
                decision = decide_with_llm_lc(parsed["text"], similar)
            elif score > 0.60:
                print("Интересная связь. Это дополнение к существующей теме.")
                decision = "complement"

                if "related_ids" not in parsed:
                    parsed["related_ids"] = []
                parsed["related_ids"].append(best_match["id"])
            else:
                decision = "new"

        if decision == "duplicate":
            print("OK: Это дубликат. Пропускаем сохранение.")
            continue
        elif decision == "complement":
            print(f"OK: Сохраняю как дополнение к {best_match['id']}")
        else:
            print("OK: Сохраняю как новую запись.")

        save_knowledge_lc(parsed, db)


if __name__ == "__main__":
    main()
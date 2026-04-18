import json
import uuid
import os
from datetime import datetime

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

# ==========================================
#               КОНФИГУРАЦИЯ
# ==========================================
LM_STUDIO_URL = "http://localhost:1234/v1"
GENERATION_MODEL = "qwen/qwen3-4b-2507"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

BRAIN_FILE = "brain.json"
FAISS_INDEX = "faiss_index"

# ==========================================
#         ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ==========================================
print("📦 Загрузка LangChain компонентов...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

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


def rebuild_faiss_index() -> FAISS:
    brain = load_brain()

    index = faiss.IndexFlatIP(768)  # 768 = размерность all-mpnet-base-v2

    # 2. Создаём пустой FAISS-объект с этим индексом
    db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

    if not brain:
        print("⚠️ brain.json пуст. Возвращаю пустой индекс.")
        return db

    # 3. Готовим данные
    texts = [item["text"] for item in brain]
    metadatas = [{"id": item["id"], "topic": item.get("topic", "")} for item in brain]
    ids = [item["id"] for item in brain]

    print("🔄 Векторизация + добавление в индекс...")

    # 4. Векторизуем ВРУЧНУЮ (чтобы контролировать нормализацию)
    vectors = embeddings.embed_documents(texts)
    vectors_np = np.array(vectors, dtype=np.float32)

    faiss.normalize_L2(vectors_np)

    # 6. Добавляем векторы в индекс напрямую (быстро, без лишней абстракции)
    db.index.add(vectors_np)

    # 7. Синхронизируем docstore и index_to_docstore_id (это делает LangChain внутри add_texts, но мы делаем вручную для контроля)
    from langchain_core.documents import Document
    for i, (text, meta, uid) in enumerate(zip(texts, metadatas, ids)):
        db.docstore._dict[uid] = Document(page_content=text, metadata=meta)
        db.index_to_docstore_id[i] = uid

    db.save_local(FAISS_INDEX)
    print("✅ Индекс перестроен (Cosine/IP, dim=768)")
    return db


def find_similar_lc(query_text: str, db: FAISS, threshold: float = 0.75, top_k: int = 5) -> list:
    if db.index.ntotal == 0:
        print("⚠️ Индекс пуст. Поиск невозможен.")
        return []

    actual_k = min(top_k, db.index.ntotal)
    docs_with_scores = db.similarity_search_with_score(query_text, k=actual_k)

    print(f"\nТОП-{len(docs_with_scores)} наиболее релевантных:")
    similar = []
    for doc, score in docs_with_scores:
        preview = doc.page_content[:60].replace('\n', ' ').strip()
        print(f"  • [{score:.3f}] {preview}...")

        if score > threshold:
            similar.append({
                "id": doc.metadata.get("id"),
                "similarity": score,
                "text": doc.page_content
            })
    print("-" * 50)
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

    print(f"OK: Сохранено: {entry['text'][:40]}...")
    return entry_id


# ==========================================
#                   main
# ==========================================

def main():
    print("🧠 BrAIn Ingest (LangChain Edition)")
    print("Вводи текст (или 'q' для выхода)")
    print("-" * 30)

    # Загружаем векторы
    if not os.path.exists(FAISS_INDEX):
        db = rebuild_faiss_index()
    else:
        db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
        print("OK: Существующий FAISS-индекс загружен.")

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

        similar = find_similar_lc(parsed["text"], db)

        if not similar:
            print("OK: Похожих тем не найдено.")
            decision = "new"
        else:
            best_match = similar[0]
            score = best_match['similarity']
            print(f"Прикреплено к: {best_match['text'][:30]}... (сходство: {score:.3f})")
            print("=" * 50)


            # Логика зон
            if score > 0.90:
                print("Высокое сходство. Проверяю на дубликат...")
                decision = decide_with_llm_lc(parsed["text"], similar)
            elif score > 0.75:
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
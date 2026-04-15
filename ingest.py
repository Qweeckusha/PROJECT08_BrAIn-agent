import json
import uuid
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

# === КОНФИГ ===
API_URL = "http://localhost:1234/v1/chat/completions"
GENERATION_MODEL = "qwen/qwen3-4b-2507"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

KNOWLEDGE_FILE = "brain.json"
VECTORS_FILE = "vectors.json"


# === СИСТЕМНЫЙ ПРОМПТ ===
SYSTEM_PROMPT = """Ты — парсер знаний. Извлеки информацию из сообщения пользователя.
Верни ТОЛЬКО валидный JSON без markdown и пояснений. Если информации нет, то верни `None` без JSON, только одно слово None:
{
  "text": "основная мысль, можно структурировать текст, но не удалять важные детали",
  "topic": "общая тема, например: [ML.classification.RandomForest, Classification.RandomForest.DecisionTree, Math], но кратко, максимальная глубина уточнения 2 и через точку",
  "subtopic": "Здесь можно уточнить тему с русским и английским например: [Метод классификации RandomForest в ML]",
  "tags": ["tag1", "tag2", "tag3"],
  "level": "beginner|intermediate|advanced",
  "status": "draft"
}"""


# === ИНИЦИАЛИЗАЦИЯ ===
print(f"📦 Загрузка эмбеддинга {EMBEDDING_MODEL_NAME}...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
print("✅ Эмбеддер готов!")
print("=== BrAIn Ingest Module готов ===")


# === ФУНКЦИИ ===

def call_llm(user_input: str) -> dict:
    """Вызов локальной модели для парсинга"""
    try:
        resp = requests.post(API_URL, json={
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.5,
            "max_tokens": 1024
        })
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # Чистка от markdown ```json ... ```
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
        return None

def get_embedding(text: str) -> list:
    """Генерация вектора через sentence-transformers"""
    return embedder.encode(text).tolist()

def get_text_by_id(entry_id: str) -> str:
    """Быстрый поиск текста по ID"""
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
        for entry in db:
            if entry["id"] == entry_id:
                return entry["text"]
    except:
        pass
    return ""

def find_similar(query_vector: list, threshold: float = 0.85) -> list:
    """Находит записи с косинусной близостью > threshold"""
    try:
        with open(VECTORS_FILE, "r", encoding="utf-8") as f:
            vectors = json.load(f)
    except FileNotFoundError:
        return []

    if not vectors:
        return []

    query_vec = np.array(query_vector)
    similar = []

    for item in vectors:
        db_vec = np.array(item["vector"])
        cos_sim = np.dot(query_vec, db_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(db_vec))

        if cos_sim > threshold:
            similar.append({
                "id": item["id"],
                "similarity": float(cos_sim),
                "text": get_text_by_id(item["id"])
            })

    return sorted(similar, key=lambda x: x["similarity"], reverse=True)

def decide_with_llm(new_text: str, existing_entries: list) -> str:
    """
    Эта функция валидирует входящие тексты, основная задача - это обнаружение деталей, которые могут выступать в качестве дополнений
    """
    # Формируем чистый контекст
    context = "\n".join([f"- {e['text']}" for e in existing_entries])

    prompt = f"""Ты — редактор базы знаний.
Сравни НОВЫЙ текст с СУЩЕСТВУЮЩИМИ записями.

СУЩЕСТВУЮЩИЕ ЗАПИСИ:
{context}

НОВЫЙ ТЕКСТ:
"{new_text}"

Задача:
Если НОВЫЙ текст просто перефразирует существующий (смысл тот же) -> 'duplicate'
Если НОВЫЙ текст добавляет детали, уточнения или факты, которых нет в старых -> 'complement'
Если НОВЫЙ текст о совершенно другом -> 'new'

Ответь ТОЛЬКО одним словом."""

    try:
        resp = requests.post(API_URL, json={
            "model": GENERATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1, # Отвечать со строгость
            "max_tokens": 10
        })
        decision = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return decision if decision in ["duplicate", "complement", "new"] else "new"
    except:
        return "new"

def save_knowledge(parsed: dict, vector: list):
    entry_id = str(uuid.uuid4())

    # Берем related_ids из parsed, если они там есть, иначе пустой список
    related = parsed.get("related_ids", [])

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

    vector_entry = {"id": entry_id, "vector": vector}

    # Сохраняем knowledge
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            db = json.load(f)
    except FileNotFoundError:
        db = []
    db.append(entry)
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    # Сохраняем vectors
    try:
        with open(VECTORS_FILE, "r", encoding="utf-8") as f:
            vectors = json.load(f)
    except FileNotFoundError:
        vectors = []
    vectors.append(vector_entry)
    with open(VECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)

    print(f"✅ Сохранено: {entry['text'][:40]}...")

# ==========================================
# 🚀 ГЛАВНЫЙ ЦИКЛ
# ==========================================

def main():
    print("🧠 BrAIn: Режим Записи")
    print("Вводи текст (или 'q' для выхода)")
    print("-" * 30)

    while True:
        user_input = input("\n📝 Ты: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue

        print("⏳ Парсю смысл...")
        parsed = call_llm(user_input)

        if not parsed or not all(k in parsed for k in ["text", "topic", "tags"]):
            print("❌ Не удалось распарсить JSON. Попробуй четче.")
            print(parsed)
            continue

        print("🔢 Векторизую...")
        vector = get_embedding(parsed["text"])

        print(" Проверяю на связи...")
        # 1. Ищем ПОХОЖЕЕ (порог ниже, чтобы ловить связи тем)
        similar = find_similar(vector, threshold=0.60)

        if not similar:
            print("✨ Новое: Похожих тем не найдено.")
            decision = "new"
        else:
            best_match = similar[0]
            score = best_match['similarity']
            print(f"🔗 Найдено: {best_match['text'][:50]}... (сходство: {score:.2f})")

            # 2. Логика зон
            if score > 0.85:
                # Зона дубликатов
                print("⚠️ Высокое сходство. Проверяю на дубликат...")
                decision = decide_with_llm(parsed["text"], similar)
            elif score > 0.60:
                # Зона дополнений
                print("💡 Интересная связь. Это выглядит как дополнение к существующей теме.")
                decision = "complement"

                if "related_ids" not in parsed:
                    parsed["related_ids"] = []
                parsed["related_ids"].append(best_match["id"])

            else:
                # Слабая связь, считаем новым
                decision = "new"

        # 3. Действие по решению
        if decision == "duplicate":
            print("♻️ Это дубликат. Пропускаем.")
            continue
        elif decision == "complement":
            print(f"🔗 Сохраняю как дополнение к {best_match['id']}")
            # related_ids уже добавлены выше
        else:
            print("✅ Сохраняю как новую запись.")

        save_knowledge(parsed, vector)

if __name__ == "__main__":
    main()
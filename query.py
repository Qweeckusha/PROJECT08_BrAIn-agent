import numpy as np
import json
import requests
from sentence_transformers import SentenceTransformer

# === КОНФИГ ===
# Для ответов лучше использовать Thinking-модель, она умнее связывает факты
API_URL = "http://localhost:1234/v1/chat/completions"
GENERATION_MODEL = "qwen/qwen3-4b-2507"

# Эмбеддер должен быть ТОТ ЖЕ, что и в ingest.py!
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

KNOWLEDGE_FILE = "brain.json"
VECTORS_FILE = "vectors.json"

SYSTEM_PROMPT = """Ты — опытный наставник по Machine Learning. 
Твоя задача — объяснять концепции простым языком, опираясь СТРОГО на предоставленный контекст. Записи из контекста нужно будет связать между собой, если это возможно.

ПРАВИЛА ОТВЕТА:
1. Объясняй своими словами, не копируй текст дословно.
2. Используй аналогии и примеры, если они уместны.
3. Структурируй ответ: короткое определение → суть → пример/применение.
4. Если в контексте нет ответа, честно скажи: "В базе пока нет информации об этом".
5. Не выдумывай факты и не добавляй знания извне.
6. Отвечай кратко (3-7 предложений), если не запрошено подробное объяснение.

СТОП-СЛОВА:
- Не начинай с фраз вроде "На основе контекста...", "В предоставленных данных...".
- Не перечисляй источники списком.
- Избегай сложных терминов без пояснений."""

# === ИНИЦИАЛИЗАЦИЯ ===
print("📦 Загрузка эмбеддера...")
# На CPU, чтобы не конфликтовать с видеопамятью модели
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
print("=== BrAIn Query Module готов ===")


# === ФУНКЦИИ ===

def get_embedding(text: str) -> list:
    """Превращает вопрос в вектор"""
    return embedder.encode(text).tolist()


def search_context(query_vector: list, top_k: int = 5) -> list:
    """
    РЕТРИВЕР: Находит top_k самых похожих записей в базе.
    """
    try:
        with open(VECTORS_FILE, "r", encoding="utf-8") as f:
            vectors = json.load(f)
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            knowledge = json.load(f)
    except FileNotFoundError:
        return []

    if not vectors:
        return []

    query_arr = np.array(query_vector)
    results = []

    # 1. Сравниваем вопрос со всеми записями
    for item in vectors:
        db_vec = np.array(item["vector"])

        # Косинусное сходство
        similarity = np.dot(query_arr, db_vec) / (np.linalg.norm(query_arr) * np.linalg.norm(db_vec))

        # Ищем текст по ID
        text = ""
        for k in knowledge:
            if k["id"] == item["id"]:
                text = k["text"]
                break

        results.append({
            "id": item["id"],
            "text": text,
            "similarity": float(similarity)
        })

    # 2. Сортируем и берем лучших
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# === ДИНАМИЧЕСКИЙ ПРОМПТ (формируется каждый запрос) ===
def generate_answer(question: str, context_list: list) -> str:

    context_text = "\n\n".join([f"• {item['text']}" for item in context_list])

    user_prompt = f"""
КОНТЕКСТ (база знаний):
{context_text}

ВОПРОС:
{question}

Ответ:
"""

    try:
        resp = requests.post(API_URL, json={
            "model": GENERATION_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 1350,
            "repeat_penalty": 1.1
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Ошибка генерации: {e}"


# === ГЛАВНЫЙ ЦИКЛ ===

def main():
    print("🧠 Спроси меня что-нибудь (или 'q' для выхода)")
    print("-" * 30)

    while True:
        user_input = input("\n🗣 Ты: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue

        # 1. Векторизация вопроса
        print("🔍 Поиск знаний...")
        q_vec = get_embedding(user_input)

        # 2. Retriever (Поиск в базе)
        matches = search_context(q_vec)

        if not matches:
            print("🤷 Я ничего не нашел по этой теме в своей базе.")
            continue

        # Найденное (для отладки)
        print(f"📚 Найдено {len(matches)} источников:")
        for m in matches:
            print(f"   • {m['text'][:60]}... (сходство: {m['similarity']:.2f})")

        # 3. Генерация ответа
        print("🧠 Думаю над ответом...")
        answer = generate_answer(user_input, matches)
        print(f"\n💡 BrAIn: {answer}")


if __name__ == "__main__":
    main()
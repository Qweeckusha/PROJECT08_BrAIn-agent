import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    with open("brain.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ brain.json не найден! Сначала запусти ingest.py из main ветки.")
    exit()

texts = [item["text"] for item in data]
# Сохраняем ID и Topic как метаданные (они тоже сохранятся в бинарный файл)
metadatas = [{"id": item["id"], "topic": item.get("topic", "unknown")} for item in data]

print("⏳ Векторизация данных...")
db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

db.save_local("faiss_index")

print("✅ База сохранена в папку 'faiss_index'. Теперь query.py будет работать мгновенно.")
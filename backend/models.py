from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("Загрузка тяжелых моделей Embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

print(f"OK: {EMBEDDING_MODEL_NAME} ready")
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



# ==========================================
#                 КОМПОНЕНТЫ
# ==========================================

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Векторная база ---
if not os.path.exists("faiss_index"):
    print("Папка faiss_index не найдена.")
    exit()

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})
print("✅: База загружена из 'faiss_index'.")

# --- LLM (Через LM Studio) ---
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen/qwen3-4b-2507",
    temperature=0.5
)


# ==========================================
#            Chain и интерфейс
# ==========================================

prompt = ChatPromptTemplate.from_template("""Ты — опытный наставник по Machine Learning. 
Твоя задача — объяснять концепции простым языком, опираясь СТРОГО на предоставленный контекст. 
Записи из контекста нужно будет связать между собой, если это возможно.

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
- Избегай сложных терминов без пояснений.

КОНТЕКСТ (база знаний):
{context}

ВОПРОС:
{question}

Ответ:""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Интерфейс
print("\n💡 Спрашивай (или 'q' для выхода):")
while True:
    user_input = input("\n🗣 Ты: ").strip()
    if user_input.lower() == 'q':
        break
    if not user_input:
        continue

    print("⏳ Думаю...")
    try:
        response = chain.invoke(user_input)
        print(f"BrAIn: {response}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
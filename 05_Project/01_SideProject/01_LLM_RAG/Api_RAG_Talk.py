from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "chroma_db"

GROQ_API_KEY = "GROQ_API_KEY"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=str(db_path),
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b"
)

print("RAG 질문 답변 시스템 시작")
print("종료하려면 exit 입력\n")

while True:
    question = input("질문: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("프로그램 종료")
        break

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
너는 문서 기반 질의응답 AI이다.
아래 문서 내용만 참고해서 질문에 답변해라.
문서에 없는 내용은 모른다고 답변해라.

[문서 내용]
{context}

[질문]
{question}

[답변]
"""

    response = llm.invoke(prompt)

    print("\n답변:")
    print(response.content)
    print("-" * 50)
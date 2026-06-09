from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / "chroma_db"

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

llm = ChatOllama(
    model="qwen3:4b",
    temperature=0.2
)

chat_history = []

print("로컬 RAG 챗봇 시작")
print("종료하려면 exit, quit, q 입력\n")

while True:
    question = input("질문: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("프로그램 종료")
        break

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])
    history_text = "\n".join(chat_history[-10:])

    prompt = f"""
당신은 문서 기반 질의응답 AI입니다.

규칙:
1. 반드시 제공된 문서를 우선적으로 참고합니다.
2. 문서에 없는 내용은 "문서에서 확인할 수 없습니다."라고 답합니다.
3. 이전 대화를 참고하여 자연스럽게 답합니다.
4. 추론 과정은 출력하지 않습니다.
5. 최종 답변만 한국어로 출력합니다.

[이전 대화]
{history_text}

[문서 내용]
{context}

[사용자 질문]
{question}

[답변]
"""

    response = llm.invoke(prompt)

    answer = response.content

    print("\n답변:")
    print(answer)
    print("-" * 50)

    chat_history.append(f"사용자: {question}")
    chat_history.append(f"AI: {answer}")
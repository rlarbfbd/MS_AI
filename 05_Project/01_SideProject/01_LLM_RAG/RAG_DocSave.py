from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "C:/Users/Admin/OneDrive/Desktop/MS_AI/Git/05_Project/01_SideProject/simple.txt"
db_path = BASE_DIR / "chroma_db"

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(db_path)
)

print("문서 저장 완료")
print(f"저장된 조각 수: {len(chunks)}")
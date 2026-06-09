from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

#BASE_DIR = Path(__file__).resolve().parent
#file_path = BASE_DIR / "C:/Users/Admin/OneDrive/Desktop/MS_AI/Git/05_Project/01_SideProject/simple.txt"
file_path = r"C:\Users\Admin\OneDrive\Desktop\MS_AI\Git\05_Project\01_SideProject\simple.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

sentences = re.split(r"(?<=[.!?。！？\n])\s+", text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

print("검색 기반 AI 시작")
print("종료하려면 exit 입력\n")

while True:
    question = input("질문: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("프로그램 종료")
        break

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, sentence_vectors)[0]

    top_indices = similarities.argsort()[::-1][:3]

    print("\n관련 내용:")

    found = False

    for idx in top_indices:
        if similarities[idx] > 0:
            print(f"- {sentences[idx]}")
            found = True

    if not found:
        print("관련 내용을 찾지 못했습니다.")

    print("-" * 50)
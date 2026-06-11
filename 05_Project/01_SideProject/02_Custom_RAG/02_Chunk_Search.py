# RAG라기보다는 문서 검색에 가까움
# 구조: 질문 -> 유사한 청크 찾기 -> 청크 그대로 출력

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = r"C:\Users\Admin\OneDrive\Desktop\MS_AI\Git\05_Project\01_SideProject\simple.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()


def split_by_markdown_headings(text):
    chunks = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks


chunks = split_by_markdown_headings(text)

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

print("청크 기반 검색 AI 시작")
print("종료하려면 exit 입력\n")

while True:
    question = input("질문: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("프로그램 종료")
        break

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    top_indices = similarities.argsort()[::-1][:1]

    print("\n관련 문서 청크:")

    found = False

    for idx in top_indices:
        if similarities[idx] > 0:
            print(f"\n[유사도: {similarities[idx]:.4f}]")
            print(chunks[idx])
            found = True

    if not found:
        print("관련 내용을 찾지 못했습니다.")

    print("-" * 50)
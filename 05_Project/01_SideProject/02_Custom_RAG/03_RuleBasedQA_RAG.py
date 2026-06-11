# 구조: 질문 -> 유사한 청크 여러 개 찾기 -> 청크에서 답 찾기 -> 답변만 출력

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


def extract_answer(question, chunks):
    question = question.strip()
    question_no_space = question.replace(" ", "")

    keywords = re.findall(r"[가-힣a-zA-Z0-9]+", question)
    keywords = [word for word in keywords if len(word) >= 2]

    matched_title_chunks = []
    matched_lines = []

    for chunk in chunks:
        lines = chunk.split("\n")

        # 1. 제목 매칭 먼저 검사
        title = lines[0].strip()
        title_no_space = title.replace(" ", "")

        if title.startswith("#"):
            if question in title or question_no_space in title_no_space:
                matched_title_chunks.append(chunk)
                continue

        # 2. 일반 줄 매칭
        for line in lines:
            line_clean = line.strip()

            if not line_clean or line_clean.startswith("#"):
                continue

            line_no_space = line_clean.replace(" ", "")

            if question in line_clean or question_no_space in line_no_space:
                matched_lines.append(line_clean)
                continue

            for keyword in keywords:
                if keyword in line_clean:
                    matched_lines.append(line_clean)
                    break

    matched_title_chunks = list(dict.fromkeys(matched_title_chunks))
    matched_lines = list(dict.fromkeys(matched_lines))

    if matched_title_chunks:
        return "\n\n".join(matched_title_chunks)

    if matched_lines:
        return "\n".join(matched_lines)

    return "관련 내용을 찾지 못했습니다."


chunks = split_by_markdown_headings(text)

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4)
)

chunk_vectors = vectorizer.fit_transform(chunks)

print("청크 기반 검색 AI 시작")
print("종료하려면 exit 입력\n")

while True:
    question = input("질문: ").strip()

    if question.lower() in ["exit", "quit", "q"]:
        print("프로그램 종료")
        break

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    top_k = 5
    top_indices = similarities.argsort()[::-1][:top_k]

    selected_chunks = [
        chunks[idx]
        for idx in top_indices
        if similarities[idx] > 0
    ]

    print("\n답변:")

    if selected_chunks:
        answer = extract_answer(question, selected_chunks)
        print(answer)
    else:
        print("관련 내용을 찾지 못했습니다.")

    print("-" * 50)
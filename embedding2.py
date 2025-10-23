import google.generativeai as genai
import pickle
import os

# Gemini API 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 부동산 정보 청크
article_chunks = [
    """목동7단지 재건축 분담금은 아직 확정되지 않았지만, 언론과 추진위 기준 추정치를 보면 다음과 같습니다: 전용 59㎡ → 59㎡ 유지 시: 약 2억 1,305만 원 환급 가능성 있음. 
    전용 89㎡ → 84㎡ 변경 시: 약 2억 7,562만 원 환급 예상. 
    전용 101㎡ → 59㎡ 축소 시: 최대 11억 2,198만 원 환급 가능성 언급됨. 
    또 다른 추정에서는 27평형 보유자가 84㎡ 신청 시 약 4,517만 원 환급 가능성 제시됨. 
    요약하자면, 많은 경우 환급금이 발생할 가능성이 거론되고 있으나, 사업비 상승과 기간 지연 등이 변수로 작용할 수 있습니다."""
]

# Gemini를 사용한 임베딩 생성
chunk_embeddings = []
embedding_model = "text-embedding-004"

for chunk in article_chunks:
    result = genai.embed_content(
        model=embedding_model,
        content=chunk,
        task_type="retrieval_document"
    )
    chunk_embeddings.append(result["embedding"])

# 임베딩 저장
with open("embeddings.pkl", "wb") as f:
    pickle.dump({"chunks": article_chunks, "embeddings": chunk_embeddings}, f)
    
print(f"총 {len(article_chunks)}개의 청크가 Gemini로 임베딩되었습니다.")
print(f"임베딩 차원: {len(chunk_embeddings[0])}")

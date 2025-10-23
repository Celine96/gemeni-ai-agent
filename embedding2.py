import pickle
import os
import google.generativeai as genai
from typing import List

# Gemini API 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def create_embeddings_gemini(chunks: List[str], output_file: str = "embeddings.pkl"):
    """
    Gemini API를 사용하여 텍스트 청크의 임베딩을 생성하고 저장합니다.
    
    Args:
        chunks: 임베딩할 텍스트 청크 리스트
        output_file: 저장할 pickle 파일 경로
    """
    print(f"총 {len(chunks)}개의 청크에 대한 임베딩을 생성합니다...")
    
    embeddings = []
    embedding_model = 'models/text-embedding-004'
    
    for idx, chunk in enumerate(chunks):
        try:
            # Gemini Embedding API 호출
            result = genai.embed_content(
                model=embedding_model,
                content=chunk,
                task_type="retrieval_document"  # 문서 임베딩용
            )
            embeddings.append(result['embedding'])
            
            if (idx + 1) % 10 == 0:
                print(f"진행 중... {idx + 1}/{len(chunks)}")
                
        except Exception as e:
            print(f"오류 발생 (청크 {idx}): {e}")
            embeddings.append([0] * 768)  # 오류 시 빈 임베딩
    
    # 저장
    data = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"✅ 임베딩이 {output_file}에 저장되었습니다.")
    return embeddings

if __name__ == "__main__":
    # 예시: 기존 embeddings.pkl을 로드하여 Gemini로 재생성
    try:
        with open("embeddings.pkl", "rb") as f:
            old_data = pickle.load(f)
            chunks = old_data["chunks"]
        
        print("기존 청크를 발견했습니다. Gemini 임베딩으로 재생성합니다...")
        create_embeddings_gemini(chunks, "embeddings_gemini.pkl")
        
    except FileNotFoundError:
        print("embeddings.pkl을 찾을 수 없습니다.")
        print("새로운 청크를 생성하려면 아래와 같이 사용하세요:")
        print("""
# 예시
sample_chunks = [
    "부동산 양도소득세는 부동산을 팔았을 때 발생하는 소득에 대해 부과되는 세금입니다.",
    "취득세는 부동산을 취득할 때 납부하는 세금으로, 취득가액을 기준으로 계산됩니다."
]
create_embeddings_gemini(sample_chunks)
        """)

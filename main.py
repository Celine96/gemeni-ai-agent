import logging
import os
import numpy as np
import pickle

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Kakao Chatbot with Gemini API", "version": "1.0"}

# Gemini API 설정
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables!")
    else:
        genai.configure(api_key=api_key)
        logger.info("Gemini API configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")

# Gemini 모델 초기화
chat_model = genai.GenerativeModel('gemini-1.5-pro')
embedding_model = 'models/text-embedding-004'

# Define Pydantic models for nested JSON structure
class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

@app.post("/generate")
async def generate_text(request: RequestBody):
    """기본 텍스트 생성 엔드포인트"""
    prompt = request.action.params.get("prompt")
    
    if not prompt:
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "질문을 입력해주세요."
                        }
                    }
                ]
            }
        }
    
    try:
        logger.info(f"Generating response for: {prompt[:50]}...")
        
        # Gemini API 호출
        response = chat_model.generate_content(prompt)
        
        logger.info("Response generated successfully")
        
        # Return the generated text
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response.text
                        }
                    }
                ]
            }
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                        }
                    }
                ]
            }
        }

## Embeddings 로드
article_chunks = []
chunk_embeddings = []

try:
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        article_chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
    logger.info(f"✅ Embeddings loaded successfully ({len(article_chunks)} chunks)")
except FileNotFoundError:
    logger.warning("⚠️ embeddings.pkl not found. /custom endpoint will work without RAG.")
except Exception as e:
    logger.error(f"❌ Error loading embeddings: {e}")

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    from numpy import dot
    from numpy.linalg import norm
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot(a, b) / (norm_a * norm_b)

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """RAG 기반 커스텀 응답 생성 엔드포인트"""
    try:
        # Extract prompt from nested JSON
        prompt = request.action.params.get("prompt")
        
        if not prompt:
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": "질문을 입력해주세요."
                            }
                        }
                    ]
                }
            }
        
        logger.info(f"Processing custom request: {prompt[:50]}...")
        
        # embeddings.pkl이 있는 경우에만 RAG 사용
        if article_chunks and chunk_embeddings:
            try:
                # Gemini Embedding API 호출
                result = genai.embed_content(
                    model=embedding_model,
                    content=prompt,
                    task_type="retrieval_query"
                )
                q_embedding = result['embedding']
                
                # 코사인 유사도 계산
                similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
                
                # 가장 유사한 청크 2개 선택
                top_n = 2
                top_indices = np.argsort(similarities)[-top_n:][::-1]
                selected_context = "\n\n".join([article_chunks[i] for i in top_indices])
                
                logger.info(f"RAG context retrieved (similarity: {similarities[top_indices[0]]:.3f})")
                
            except Exception as e:
                logger.error(f"Error in RAG processing: {e}")
                selected_context = ""
        else:
            selected_context = ""
            logger.info("No embeddings available, using direct generation")
        
        # Gemini에게 전달할 프롬프트 구성
        if selected_context:
            query = f"""아래 컨텍스트를 사용하여 질문에 답변하세요.

당신은 REXA로, 10년 경력의 부동산 전문가입니다. 세금(양도소득세, 재산세, 증여/상속세, 취득세), 경매, 민법, 건축법에 정통한 챗봇입니다.
전문 상담사처럼 정중하고 신뢰감 있는 어조로 답변하세요. 빠른 응답을 위해 답변은 250토큰 이하로 유지하세요.
정보를 모르는 경우, 사용자에게 한 번 더 질문하세요.

컨텍스트:
\"\"\"
{selected_context}
\"\"\"

질문: {prompt}

답변은 반드시 한국어로 작성하세요.
"""
        else:
            query = f"""당신은 REXA로, 10년 경력의 부동산 전문가입니다. 세금(양도소득세, 재산세, 증여/상속세, 취득세), 경매, 민법, 건축법에 정통한 챗봇입니다.
전문 상담사처럼 정중하고 신뢰감 있는 어조로 답변하세요. 빠른 응답을 위해 답변은 250토큰 이하로 유지하세요.
정보를 모르는 경우, 사용자에게 한 번 더 질문하세요.

질문: {prompt}

답변은 반드시 한국어로 작성하세요.
"""
        
        # Gemini API 호출
        response = chat_model.generate_content(
            query,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
            )
        )
        
        logger.info("Custom response generated successfully")
        
        # Return the generated text
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response.text
                        }
                    }
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in /custom endpoint: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
                        }
                    }
                ]
            }
        }

@app.get("/health")
def health_check():
    """헬스 체크 엔드포인트"""
    status = {
        "status": "healthy",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "embeddings_loaded": len(article_chunks) > 0,
        "chunks_count": len(article_chunks)
    }
    return status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

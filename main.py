import logging
import os
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World - Gemini Version"}

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    logger.info("Gemini API configured successfully")
else:
    logger.error("GOOGLE_API_KEY not found!")

# Define Pydantic models for nested JSON structure
class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

# 모델 설정
GENERATION_MODEL = "gemini-2.5-flash"  # 빠르고 효율적
EMBEDDING_MODEL = "models/embedding-001"

# 유사도 임계값 (이 값보다 낮으면 일반 대화 모드)
SIMILARITY_THRESHOLD = 0.3

@app.post("/generate")
async def generate_text(request: RequestBody):
    """기본 텍스트 생성 엔드포인트"""
    prompt = request.action.params.get("prompt")
    logger.info(f"[/generate] Received: {prompt}")
    
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=4.0
        )
        
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
    except asyncio.TimeoutError:
        logger.error("[/generate] Timeout")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "응답 시간이 초과되었습니다. 다시 시도해주세요."
                        }
                    }
                ]
            }
        }
    except Exception as e:
        logger.error(f"[/generate] Error: {e}")
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

## Embeddings with Gemini
import pickle

try:
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        article_chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
    logger.info(f"Embeddings loaded: {len(article_chunks)} chunks")
except FileNotFoundError:
    logger.error("embeddings.pkl not found! RAG disabled")
    article_chunks = []
    chunk_embeddings = []

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """하이브리드 RAG + 일반 대화 엔드포인트"""
    prompt = request.action.params.get("prompt")
    logger.info(f"[/custom] Received: {prompt}")
    
    try:
        # RAG 사용 가능 여부 확인
        use_rag = False
        max_similarity = 0.0
        selected_context = ""
        
        if article_chunks and chunk_embeddings:
            # 질문 임베딩 생성
            logger.info("[/custom] Generating embedding...")
            q_embedding = await asyncio.wait_for(
                asyncio.to_thread(
                    genai.embed_content,
                    model=EMBEDDING_MODEL,
                    content=prompt,
                    task_type="retrieval_query"
                ),
                timeout=2.0
            )
            q_embedding = q_embedding["embedding"]
            
            # 유사도 계산
            similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
            max_similarity = max(similarities) if similarities else 0.0
            
            logger.info(f"[/custom] Max similarity: {max_similarity:.3f}")
            
            # 임계값 체크
            if max_similarity >= SIMILARITY_THRESHOLD:
                use_rag = True
                top_n = 2
                top_indices = np.argsort(similarities)[-top_n:][::-1]
                selected_context = "\n\n".join([article_chunks[i] for i in top_indices])
                logger.info("[/custom] Using RAG mode")
            else:
                logger.info("[/custom] Using general conversation mode")
        
        # 프롬프트 구성
        if use_rag:
            # RAG 모드: 부동산 데이터 사용
            query = f"""Use the below context to answer the question. 
You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would. 
Keep your answers under 250 tokens.

Context:
\"\"\"
{selected_context}
\"\"\"

Question: {prompt}

Respond in Korean in a professional and helpful manner.
"""
            temperature = 0.0
        else:
            # 일반 대화 모드: Gemini 자체 지식 사용
            query = f"""You are REXA, a friendly and knowledgeable real estate expert chatbot with 10 years of experience.

The user said: "{prompt}"

Instructions:
- If this is a greeting or casual conversation, respond naturally and warmly
- If this is a simple question (like math), answer it directly and briefly
- If this is about real estate topics not in your specific knowledge base, politely explain you don't have detailed information on that specific topic, but offer to help with general real estate advice
- Always maintain a professional yet friendly tone
- Keep your response under 200 tokens
- ALWAYS respond in Korean

Respond naturally in Korean.
"""
            temperature = 0.7
        
        logger.info(f"[/custom] Generating response (temp={temperature})...")
        
        # Gemini 응답 생성
        model = genai.GenerativeModel(
            GENERATION_MODEL,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 512,
            }
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, query),
            timeout=4.0
        )
        
        logger.info("[/custom] Success")
        
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
        
    except asyncio.TimeoutError:
        logger.error("[/custom] Timeout")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "응답 생성에 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해주세요."
                        }
                    }
                ]
            }
        }
    except Exception as e:
        logger.error(f"[/custom] Error: {type(e).__name__}: {str(e)}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 한번 질문해주시겠어요?"
                        }
                    }
                ]
            }
        }

# Health check
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunks_loaded": len(article_chunks),
        "rag_enabled": len(article_chunks) > 0
    }

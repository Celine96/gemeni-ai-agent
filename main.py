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
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"  # embeddings.pkl과 동일한 모델 사용

# 유사도 임계값
SIMILARITY_THRESHOLD = 0.3

# 안전 설정 (차단 완화)
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

@app.post("/generate")
async def generate_text(request: RequestBody):
    """기본 텍스트 생성 엔드포인트"""
    prompt = request.action.params.get("prompt")
    logger.info(f"[/generate] Received: {prompt}")
    
    try:
        model = genai.GenerativeModel(
            GENERATION_MODEL,
            safety_settings=safety_settings
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=4.0
        )
        
        # 안전하게 텍스트 추출
        if response.candidates and response.candidates[0].content.parts:
            answer = response.text
        else:
            logger.warning(f"[/generate] No valid response: {response.prompt_feedback}")
            answer = "죄송합니다. 응답을 생성할 수 없습니다. 다른 방식으로 질문해주시겠어요?"
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
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
                            "text": "죄송합니다. 일시적인 오류가 발생했습니다."
                        }
                    }
                ]
            }
        }

## Embeddings
import pickle

try:
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        article_chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
    logger.info(f"Embeddings loaded: {len(article_chunks)} chunks, dim={len(chunk_embeddings[0])}")
except FileNotFoundError:
    logger.error("embeddings.pkl not found!")
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
        # RAG 체크
        use_rag = False
        max_similarity = 0.0
        selected_context = ""
        
        if article_chunks and chunk_embeddings:
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
            
            similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
            max_similarity = max(similarities) if similarities else 0.0
            
            logger.info(f"[/custom] Max similarity: {max_similarity:.3f}")
            
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
            query = f"""Based on the context below, answer the user's question professionally.

Context:
{selected_context}

Question: {prompt}

Provide a helpful answer in Korean. If the context doesn't contain the exact information, use what's available and note any limitations."""
            temperature = 0.0
        else:
            query = f"""You are REXA, a friendly real estate expert assistant.

User question: {prompt}

Respond naturally in Korean. Be helpful, professional, and conversational."""
            temperature = 0.7
        
        logger.info(f"[/custom] Generating response (temp={temperature})...")
        
        model = genai.GenerativeModel(
            GENERATION_MODEL,
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 512,
            },
            safety_settings=safety_settings
        )
        
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, query),
            timeout=4.0
        )
        
        # 안전하게 응답 추출
        if response.candidates and response.candidates[0].content.parts:
            answer = response.text
            logger.info("[/custom] Success")
        else:
            logger.warning(f"[/custom] Blocked: {response.prompt_feedback}")
            answer = "죄송합니다. 해당 질문에 대한 응답을 생성할 수 없습니다. 다른 방식으로 질문해주시겠어요?"
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
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
                            "text": "응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."
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
                            "text": "일시적인 오류가 발생했습니다. 다시 시도해주세요."
                        }
                    }
                ]
            }
        }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunks_loaded": len(article_chunks),
        "rag_enabled": len(article_chunks) > 0
    }

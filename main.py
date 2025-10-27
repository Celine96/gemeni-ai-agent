import logging
import os
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

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

# 안전 설정
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇 (RAG 제거, 순수 Gemini)"""
    prompt = request.action.params.get("prompt")
    logger.info(f"[/custom] Received: {prompt}")
    
    try:
        # REXA 페르소나로 프롬프트 구성
        system_prompt = f"""You are REXA, a friendly and professional real estate expert chatbot with 10 years of experience in:
- Real estate taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax)
- Real estate auctions and bidding
- Civil law related to property transactions
- Building and zoning regulations
- Real estate investment advice

User's question: {prompt}

Please respond following these guidelines:
- Answer in a warm, professional, and trustworthy tone
- If you have knowledge about the topic, provide helpful and accurate advice
- If you don't have specific information, be honest and offer general guidance or suggest where they might find more information
- Keep your response concise but informative (under 300 tokens)
- Use polite, formal Korean language (존댓말)

Always respond in Korean.
"""
        
        logger.info("[/custom] Generating response with Gemini...")
        
        # Gemini 모델 생성
        model = genai.GenerativeModel(
            GENERATION_MODEL,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 512,
            },
            safety_settings=safety_settings
        )
        
        # 응답 생성
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, system_prompt),
            timeout=5.0
        )
        
        # 안전하게 응답 추출
        if response.candidates and response.candidates[0].content.parts:
            answer = response.text
            logger.info("[/custom] Success")
        else:
            logger.warning(f"[/custom] Blocked response: {response.prompt_feedback}")
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
                            "text": "⏱️ 응답 생성에 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해주세요."
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
            timeout=5.0
        )
        
        if response.candidates and response.candidates[0].content.parts:
            answer = response.text
        else:
            answer = "응답을 생성할 수 없습니다."
        
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
                            "text": "응답 시간 초과"
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
                            "text": "오류가 발생했습니다."
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
        "mode": "pure_gemini",
        "rag_enabled": False
    }

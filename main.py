import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "REXA - Real Estate Expert Assistant"}

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found!")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)
logger.info("Gemini API configured successfully")

# Define Pydantic models for nested JSON structure
class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

# 안전 설정
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇"""
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt")
    
    try:
        # REXA 페르소나로 메시지 구성
        query = f"""You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would. 
To ensure fast responses, keep your answers under 250 tokens.

User's question: {prompt}

Please respond in Korean following the above format. If you don't have specific information about the topic, politely explain that and offer general real estate advice or guidance on where to find more information.
"""
        
        logger.info(f"[/custom] User prompt: {prompt}")
        
        # Gemini 모델 호출
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 512,
            },
            safety_settings=safety_settings
        )
        
        response = model.generate_content(query)
        
        # 응답 검증
        if response.candidates and response.candidates[0].content.parts:
            answer = response.text
            logger.info("[/custom] Response generated successfully")
        else:
            logger.warning(f"[/custom] Blocked response: {response.prompt_feedback}")
            answer = "죄송합니다. 해당 질문에 대한 응답을 생성할 수 없습니다. 다른 방식으로 질문해주시겠어요?"
        
        # Return the generated text in KakaoTalk format
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
        
    except Exception as e:
        logger.error(f"[/custom] Error: {type(e).__name__}: {e}")
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

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "gemini-2.5-flash",
        "mode": "rexa_chatbot"
    }

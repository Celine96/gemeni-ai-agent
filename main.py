import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

from google.generativeai.types import HarmCategory, HarmBlockThreshold

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

# 안전 설정 (더 강력하게)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇"""
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt")
    
    try:
        # 더 안전한 프롬프트 구조
        query = f"""You are REXA, a professional real estate advisor with 10 years of experience.

Your expertise includes:
- Real estate taxation (capital gains, property tax, gift/inheritance tax, acquisition tax)
- Property auctions
- Civil law
- Building regulations

User question: {prompt}

Please provide a helpful answer in Korean. Keep your response under 250 tokens. Be professional and trustworthy.
"""
        
        logger.info(f"[/custom] User prompt: {prompt}")
        print(prompt)
        print(query)
        
        # Gemini 모델 호출 (GPT의 temperature=0과 동일하게)
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0,  # GPT와 동일하게 0으로
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 512,
            },
            safety_settings=safety_settings
        )
        
        response = model.generate_content(query)
        
        # 상세한 응답 디버깅
        logger.info(f"[/custom] Response received")
        logger.info(f"[/custom] Candidates: {len(response.candidates) if response.candidates else 0}")
        
        # 응답 검증
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            logger.info(f"[/custom] Finish reason: {candidate.finish_reason}")
            
            if candidate.content and candidate.content.parts:
                answer = response.text
                logger.info("[/custom] Response generated successfully")
            else:
                logger.warning(f"[/custom] No content parts")
                logger.warning(f"[/custom] Safety ratings: {candidate.safety_ratings}")
                answer = "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해주세요."
        else:
            logger.warning(f"[/custom] No candidates")
            if hasattr(response, 'prompt_feedback'):
                logger.warning(f"[/custom] Prompt feedback: {response.prompt_feedback}")
            answer = "죄송합니다. 응답을 생성할 수 없습니다. 질문을 다르게 표현해주세요."
        
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
        "model": "gemini-1.5-flash",
        "mode": "rexa_chatbot"
    }

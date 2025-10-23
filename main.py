import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World - Gemini Version"}

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    try:
        # Gemini 모델 초기화
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # 텍스트 생성
        response = model.generate_content(prompt)
        
        # 카카오톡 응답 형식으로 반환
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
        logging.error(f"Gemini API error: {e}")
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

# 기존 OpenAI 임베딩 데이터 로드
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    article_chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]

def cosine_similarity(a, b):
    """코사인 유사도 계산"""
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """RAG 기반 부동산 전문 상담 엔드포인트"""
    prompt = request.action.params.get("prompt")
    
    try:
        # Gemini를 이용한 임베딩 생성
        embedding_model = "models/text-embedding-004"
        q_embedding = genai.embed_content(
            model=embedding_model,
            content=prompt,
            task_type="retrieval_query"
        )["embedding"]
        
        # 유사도 계산
        similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
        
        # 가장 유사한 청크 2개 선택
        top_n = 2
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        selected_context = "\n\n".join([article_chunks[i] for i in top_indices])

        # Gemini에게 전달할 프롬프트 구성
        query = f"""Use the below context to answer the question. 
You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would. To ensure fast responses, keep your answers under 250 tokens. 
If you don't know about the information ask the user once more time.

Context:
\"\"\"
{selected_context}
\"\"\"

Question: {prompt}

And please respond in Korean following the above format.
"""

        print(f"User prompt: {prompt}")
        print(f"Full query: {query}")
        
        # Gemini 모델로 응답 생성
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        response = model.generate_content(query)
        
        # 카카오톡 응답 형식으로 반환
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
        logging.error(f"Error in custom endpoint: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 부동산 정보를 처리하는 중 오류가 발생했습니다. 다시 한번 질문해주시겠어요?"
                        }
                    }
                ]
            }
        }

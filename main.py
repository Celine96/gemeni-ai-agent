import logging
import os
import asyncio
from datetime import datetime
from typing import Optional, List
import uuid
from collections import deque

from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Redis for queue management
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis package not installed. Using in-memory queue.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ================================================================================
# Configuration & Global Variables
# ================================================================================

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Health Check Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 5))  # seconds
MAX_UNHEALTHY_COUNT = int(os.getenv("MAX_UNHEALTHY_COUNT", 3))

# Queue Configuration
WEBHOOK_QUEUE_NAME = "rexa:webhook_queue"
WEBHOOK_PROCESSING_QUEUE = "rexa:processing_queue"
WEBHOOK_FAILED_QUEUE = "rexa:failed_queue"
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))
QUEUE_PROCESS_INTERVAL = int(os.getenv("QUEUE_PROCESS_INTERVAL", 5))  # seconds

# Global state
redis_client: Optional[Redis] = None
server_healthy = True
unhealthy_count = 0
last_health_check = datetime.now()

# In-memory queue fallback (when Redis is not available)
in_memory_webhook_queue: deque = deque()
in_memory_processing_queue: deque = deque()
in_memory_failed_queue: deque = deque()
use_in_memory_queue = False

# ================================================================================
# Pydantic Models
# ================================================================================

class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

class QueuedRequest(BaseModel):
    request_id: str
    request_body: dict
    timestamp: str
    retry_count: int = 0
    error_message: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    model: str
    mode: str
    server_healthy: bool
    last_check: str
    redis_connected: bool
    queue_size: int
    processing_queue_size: int
    failed_queue_size: int

# ================================================================================
# Redis & Queue Management
# ================================================================================

async def init_redis():
    """Initialize Redis connection"""
    global redis_client, use_in_memory_queue
    
    if not REDIS_AVAILABLE:
        logger.warning("⚠️ Redis package not installed - using in-memory queue")
        use_in_memory_queue = True
        return
    
    try:
        redis_client = await redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        use_in_memory_queue = False
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.info("📦 Using in-memory queue as fallback")
        redis_client = None
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

async def enqueue_webhook_request(request_id: str, request_body: dict) -> bool:
    """Add webhook request to queue"""
    try:
        queued_request = QueuedRequest(
            request_id=request_id,
            request_body=request_body,
            timestamp=datetime.now().isoformat(),
            retry_count=0
        )
        
        if use_in_memory_queue:
            # Use in-memory queue
            in_memory_webhook_queue.appendleft(queued_request)
            logger.info(f"✅ Request {request_id} enqueued (in-memory)")
            return True
        
        if not redis_client:
            logger.warning("Queue not available - cannot enqueue request")
            return False
        
        # Use Redis queue
        await redis_client.lpush(
            WEBHOOK_QUEUE_NAME,
            queued_request.model_dump_json()
        )
        logger.info(f"✅ Request {request_id} enqueued (Redis)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to enqueue request: {e}")
        return False

async def dequeue_webhook_request() -> Optional[QueuedRequest]:
    """Get next webhook request from queue"""
    try:
        if use_in_memory_queue:
            # Use in-memory queue
            if len(in_memory_webhook_queue) > 0:
                request = in_memory_webhook_queue.pop()
                in_memory_processing_queue.appendleft(request)
                return request
            return None
        
        if not redis_client:
            return None
        
        # Use Redis queue
        result = await redis_client.brpoplpush(
            WEBHOOK_QUEUE_NAME,
            WEBHOOK_PROCESSING_QUEUE,
            timeout=1
        )
        
        if result:
            return QueuedRequest.model_validate_json(result)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to dequeue request: {e}")
        return None

async def complete_webhook_request(request_id: str):
    """Mark webhook request as completed"""
    try:
        if use_in_memory_queue:
            # Use in-memory queue
            for req in list(in_memory_processing_queue):
                if req.request_id == request_id:
                    in_memory_processing_queue.remove(req)
                    logger.info(f"✅ Request {request_id} completed (in-memory)")
                    return
            return
        
        if not redis_client:
            return
        
        # Use Redis
        processing_items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
        for item in processing_items:
            req = QueuedRequest.model_validate_json(item)
            if req.request_id == request_id:
                await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                logger.info(f"✅ Request {request_id} completed (Redis)")
                break
    except Exception as e:
        logger.error(f"❌ Failed to complete request: {e}")

async def fail_webhook_request(request_id: str, error_message: str):
    """Move failed request to failed queue or retry"""
    try:
        if use_in_memory_queue:
            # Use in-memory queue
            for req in list(in_memory_processing_queue):
                if req.request_id == request_id:
                    req.retry_count += 1
                    req.error_message = error_message
                    in_memory_processing_queue.remove(req)
                    
                    if req.retry_count >= MAX_RETRY_ATTEMPTS:
                        in_memory_failed_queue.appendleft(req)
                        logger.error(f"❌ Request {request_id} moved to failed queue after {req.retry_count} attempts (in-memory)")
                    else:
                        in_memory_webhook_queue.appendleft(req)
                        logger.warning(f"⚠️ Request {request_id} re-queued (attempt {req.retry_count}/{MAX_RETRY_ATTEMPTS}) (in-memory)")
                    return
            return
        
        if not redis_client:
            return
        
        # Use Redis
        processing_items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
        for item in processing_items:
            req = QueuedRequest.model_validate_json(item)
            if req.request_id == request_id:
                req.retry_count += 1
                req.error_message = error_message
                
                if req.retry_count >= MAX_RETRY_ATTEMPTS:
                    await redis_client.lpush(WEBHOOK_FAILED_QUEUE, req.model_dump_json())
                    await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                    logger.error(f"❌ Request {request_id} moved to failed queue after {req.retry_count} attempts (Redis)")
                else:
                    await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
                    await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                    logger.warning(f"⚠️ Request {request_id} re-queued (attempt {req.retry_count}/{MAX_RETRY_ATTEMPTS}) (Redis)")
                break
    except Exception as e:
        logger.error(f"❌ Failed to handle failed request: {e}")

async def get_queue_sizes() -> tuple:
    """Get sizes of all queues"""
    try:
        if use_in_memory_queue:
            # Use in-memory queue
            return (
                len(in_memory_webhook_queue),
                len(in_memory_processing_queue),
                len(in_memory_failed_queue)
            )
        
        if not redis_client:
            return 0, 0, 0
        
        # Use Redis
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
        failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
        return queue_size, processing_size, failed_size
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return 0, 0, 0

# ================================================================================
# Background Tasks
# ================================================================================

async def health_check_monitor():
    """Background task to monitor server health"""
    global server_healthy, unhealthy_count, last_health_check
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            is_healthy = await perform_health_check()
            
            if is_healthy:
                server_healthy = True
                unhealthy_count = 0
                logger.debug("✅ Health check passed")
            else:
                unhealthy_count += 1
                logger.warning(f"⚠️ Health check failed (count: {unhealthy_count}/{MAX_UNHEALTHY_COUNT})")
                
                if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                    server_healthy = False
                    logger.error("❌ Server marked as unhealthy")
            
            last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Health check monitor error: {e}")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

async def perform_health_check() -> bool:
    """Perform actual health check"""
    try:
        test_model = genai.GenerativeModel(MODEL_NAME)
        test_response = test_model.generate_content(
            "Test",
            generation_config={"max_output_tokens": 10}
        )
        
        if not test_response or not test_response.candidates:
            return False
        
        # Only check Redis if we're using it
        if redis_client and not use_in_memory_queue:
            await redis_client.ping()
        
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def queue_processor():
    """Background task to process queued webhook requests"""
    logger.info("🚀 Queue processor started")
    
    while True:
        try:
            if not server_healthy:
                logger.warning("⏸️ Queue processing paused - server unhealthy")
                await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
                continue
            
            queued_request = await dequeue_webhook_request()
            
            if queued_request:
                logger.info(f"📥 Processing queued request: {queued_request.request_id}")
                
                try:
                    result = await process_gemini_request(queued_request.request_body)
                    await complete_webhook_request(queued_request.request_id)
                    logger.info(f"✅ Queued request {queued_request.request_id} processed successfully")
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"❌ Error processing queued request: {error_msg}")
                    await fail_webhook_request(queued_request.request_id, error_msg)
            else:
                await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
                
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)

# ================================================================================
# Gemini Configuration
# ================================================================================

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found!")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

genai.configure(api_key=api_key)
logger.info("Gemini API configured successfully")

try:
    available_models = genai.list_models()
    logger.info(f"Available models: {[m.name for m in available_models][:5]}")
except Exception as e:
    logger.warning(f"Could not list models: {e}")

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

MODEL_NAME = "gemini-2.5-flash-lite"

# ================================================================================
# Helper Functions
# ================================================================================

async def process_gemini_request(request_body: dict) -> dict:
    """Process Gemini API request"""
    prompt = request_body.get("action", {}).get("params", {}).get("prompt")
    
    query = f"""You are REXA, a professional real estate advisor with 10 years of experience.

Your expertise includes:
- Real estate taxation (capital gains, property tax, gift/inheritance tax, acquisition tax)
- Property auctions
- Civil law
- Building regulations

User question: {prompt}

Please provide a helpful answer in Korean. Keep your response under 250 tokens. Be professional and trustworthy.
"""
    
    logger.info(f"Processing request with prompt: {prompt}")
    
    model = genai.GenerativeModel(
        MODEL_NAME,
        generation_config={
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 500,
        },
        safety_settings=safety_settings
    )
    
    response = model.generate_content(query)
    
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        
        if candidate.content and candidate.content.parts:
            answer = response.text
            logger.info(f"✅ Success with {MODEL_NAME}")
        else:
            logger.warning(f"No content parts. Safety ratings: {candidate.safety_ratings}")
            answer = "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해주세요."
    else:
        logger.warning("No candidates in response")
        answer = "죄송합니다. 응답을 생성할 수 없습니다. 질문을 다르게 표현해주세요."
    
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

# ================================================================================
# API Endpoints
# ================================================================================

@app.get("/")
def read_root():
    return {"Hello": "REXA - Real Estate Expert Assistant"}

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇 with queue support"""
    request_id = str(uuid.uuid4())
    
    try:
        if not server_healthy:
            logger.warning(f"⚠️ Server unhealthy - queueing request {request_id}")
            
            if await enqueue_webhook_request(request_id, request.model_dump()):
                return {
                    "version": "2.0",
                    "template": {
                        "outputs": [
                            {
                                "simpleText": {
                                    "text": "현재 서버가 일시적으로 혼잡합니다. 요청이 대기열에 추가되었으며 곧 처리됩니다. 잠시만 기다려주세요."
                                }
                            }
                        ]
                    }
                }
            else:
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
        
        result = await process_gemini_request(request.model_dump())
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in /custom endpoint: {type(e).__name__}: {e}")
        
        if await enqueue_webhook_request(request_id, request.model_dump()):
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": "요청 처리 중 오류가 발생했습니다. 요청이 대기열에 추가되었으며 재시도됩니다."
                            }
                        }
                    ]
                }
            }
        
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

@app.get("/health")
async def health_check() -> HealthStatus:
    """Enhanced health check endpoint"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return HealthStatus(
        status="healthy" if server_healthy else "unhealthy",
        model=MODEL_NAME,
        mode="rexa_chatbot",
        server_healthy=server_healthy,
        last_check=last_health_check.isoformat(),
        redis_connected=(redis_client is not None and not use_in_memory_queue),
        queue_size=queue_size,
        processing_queue_size=processing_size,
        failed_queue_size=failed_size
    )

@app.get("/health/ping")
async def health_ping():
    """Simple ping endpoint for client health checks"""
    return {
        "alive": True,
        "healthy": server_healthy,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/queue/status")
async def queue_status():
    """Get detailed queue status"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return {
        "queue_type": "in-memory" if use_in_memory_queue else "redis",
        "webhook_queue": queue_size,
        "processing_queue": processing_size,
        "failed_queue": failed_size,
        "total": queue_size + processing_size + failed_size
    }

@app.post("/queue/retry-failed")
async def retry_failed_requests():
    """Manually retry all failed requests"""
    try:
        if use_in_memory_queue:
            # Use in-memory queue
            retry_count = len(in_memory_failed_queue)
            while len(in_memory_failed_queue) > 0:
                req = in_memory_failed_queue.pop()
                req.retry_count = 0  # Reset retry count
                in_memory_webhook_queue.appendleft(req)
            
            logger.info(f"✅ Retrying {retry_count} failed requests (in-memory)")
            return {"retried": retry_count, "queue_type": "in-memory"}
        
        if not redis_client:
            return {"error": "Queue not available"}
        
        # Use Redis
        failed_items = await redis_client.lrange(WEBHOOK_FAILED_QUEUE, 0, -1)
        retry_count = 0
        
        for item in failed_items:
            req = QueuedRequest.model_validate_json(item)
            req.retry_count = 0
            await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
            retry_count += 1
        
        await redis_client.delete(WEBHOOK_FAILED_QUEUE)
        
        logger.info(f"✅ Retrying {retry_count} failed requests (Redis)")
        return {"retried": retry_count, "queue_type": "redis"}
        
    except Exception as e:
        logger.error(f"❌ Failed to retry requests: {e}")
        return {"error": str(e)}

# ================================================================================
# Startup & Shutdown Events
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting REXA server...")
    
    await init_redis()
    
    asyncio.create_task(health_check_monitor())
    asyncio.create_task(queue_processor())
    
    logger.info("✅ REXA server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down REXA server...")
    await close_redis()
    logger.info("✅ REXA server shut down successfully")

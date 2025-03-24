from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_API_KEY = "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"
if not PERPLEXITY_API_KEY:
    raise ValueError("Missing PERPLEXITY_API_KEY environment variable")

@app.post("/api/chat")
async def chat_endpoint(request: Request, messages: list[dict]):
    try:
        logger.info(f"Received request with messages: {messages}")
        
        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            "model": "sonar-deep-research",
            "messages": messages,
            "max_tokens": 2000,  # Increased token limit
            # "temperature": 0.7,
            # "stop": []
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"API Response: {result}")
        
        return {"message": result['choices'][0]['message']['content']}
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

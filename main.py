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
        
        # Add system message with instructions
        system_message = {
            "role": "system",
            "content": "你是一個保險產品比較助手，專門回答關於保險產品比較的問題。如果用戶的問題不是保險產品，請回覆：'非保險產品類別相關問題無法回答'。"
        }
        
        # Create modified messages array with system message first
        modified_messages = [system_message] + messages
        
        payload = {
            "model": "sonar-deep-research",
            "messages": modified_messages,  # Use modified messages
            "max_tokens": 2000,
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"API Response: {result}")
        
        # Replace the tags in the response
        original_content = result['choices'][0]['message']['content']
        modified_content = original_content.replace("<think>", "<產品比較助手>").replace("</think>", "</產品比較助手>")
        
        return {"message": modified_content}
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

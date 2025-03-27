from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
from dotenv import load_dotenv
from openai import OpenAI

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

DEEPSEEK_API_KEY = "sk-a96c8196a00241ee9f587cf1d1f1b99d"  # Consider moving to environment variable
if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY environment variable")


    
    


    
    
@app.post("/api/ppxty")
async def chat_endpoint(request: Request, messages: list[dict]):
    
    try:
        logger.info(f"Received request with messages: {messages}")
        
        url = "https://api.perplexity.ai/chat/completions"
        
        # Add system message with instructions
        # system_message = {
        #     "role": "system",
        #     "content": "用超級詳盡的方式比較, 例如要有基礎保單架構,核心保障差異,所有比較都是用表格形式顯示"
        # }
        
        # Create modified messages array with system message first
        # modified_messages = [system_message] + messages
        
        payload = {
            # "model": "sonar-deep-research",
            "model": "r1-1776",
            "messages": messages,  # Use modified messages
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
    
@app.post("/api/ds")
async def deepseek_endpoint(request: Request, messages: list[dict]):
    
    try:
        logger.info(f"Received DeepSeek request with messages: {messages}")
        
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        # Add system message if not already present
        # if messages and messages[0]["role"] != "system":
        #     system_message = {"role": "system", "content": "用超級詳盡的方式比較, 例如要有基礎保單架構,核心保障差異,所有比較都是用表格形式顯示"}
        #     modified_messages = [system_message] + messages
        # else:
        #     modified_messages = messages
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            # model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        result = response.choices[0].message.content
        logger.info(f"DeepSeek API Response: {result}")
        
        return {"message": result}
        
    except Exception as e:
        logger.error(f"Unexpected error in DeepSeek endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
         
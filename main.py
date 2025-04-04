from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
from contextlib import asynccontextmanager
from functools import lru_cache

load_dotenv()

IsProduction = False

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize persistent clients
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    
    # Initialize AI clients with connection pooling
    app.state.deepseek_client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        http_client=app.state.http_client
    )
    
    yield
    
    # Shutdown: Clean up resources
    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cached config to avoid repeated env lookups
@lru_cache()
def get_config():
    return {
        "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "GROK2_API_KEY": os.getenv("GROK2_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GOOGLE_CX": os.getenv("GOOGLE_CX")
    }

class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str

async def get_serpapi_search_results(query: str, http_client: httpx.AsyncClient):
    """Get search results from specified domains using SerpAPI"""
    target_sites = [
        "site:manulife.com.hk", "site:aia.com.hk", "site:prudential.com.hk",
        "site:axa.com.hk", "site:sunlife.com.hk", "site:chubb.com",
        "site:scmp.com", "site:hket.com", "site:ft.com",
        "site:moneyhero.com.hk", "site:compareasia.com", "site:policypal.com",
        "site:ia.hk", "site:sfc.hk", "site:bloomberg.com", "site:forbes.com"
    ]
    
    try:
        site_filter = f"({' OR '.join(target_sites)})"
        full_query = f"{query} {site_filter}"
        
        params = {
            "engine": "google",
            "q": full_query,
            "api_key": get_config()["SERPAPI_API_KEY"],
            "google_domain": "google.com.hk",
            "hl": "en",
            "gl": "hk",
            "num": 5
        }
        
        response = await http_client.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        
        search_context = [
            f"Title: {result.get('title', 'N/A')}\n"
            f"Snippet: {result.get('snippet', 'No description available')}\n"
            f"Link: {result.get('link', '')}"
            for result in data.get("organic_results", [])[:5]
        ]
        
        return "\n\n".join(search_context) if search_context else "No relevant results found"
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return "Search service unavailable"

@app.post("/api/ppxty")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    try:
        config = get_config()
        messages = chat_request.messages
        model = chat_request.model
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
        }
        headers = {
            "Authorization": f"Bearer {config['PERPLEXITY_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        response = await request.app.state.http_client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        original_content = result['choices'][0]['message']['content']
        modified_content = original_content.replace("<think>", "<AIM AI助手>").replace("</think>", "</AIM AI助手>")
        
        return {"message": modified_content}
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ds")
async def deepseek_endpoint(chat_request: ChatRequest, request: Request):
    try:
        messages = chat_request.messages
        model = chat_request.model
        
        response = await request.app.state.deepseek_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        
        return {"message": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"DeepSeek error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dswithsearch")
async def deepseek_with_search_endpoint(chat_request: ChatRequest, request: Request):
    latest_user_message = next(
        (msg for msg in reversed(chat_request.messages) if msg.get("role") == "user"),
        None
    )
    
    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    search_context = await get_serpapi_search_results(
        latest_user_message.get("content", ""),
        request.app.state.http_client
    )
    
    messages = []
    if search_context:
        messages.append({
            "role": "system",
            "content": f"Current web search context:\n{search_context}\n\nUse this information to supplement your response."
        })
    messages.extend(chat_request.messages)
    
    try:
        response = await request.app.state.deepseek_client.chat.completions.create(
            model=chat_request.model,
            messages=messages,
            stream=False
        )
        
        return {"message": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"DeepSeek error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI service error")
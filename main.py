#edit from Lenovo
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import logging
from dotenv import load_dotenv
from openai import OpenAI
import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import asyncio

# Load environment variables from .env file
load_dotenv()

# Set production flag (assumed False for development; adjust as needed)
IsProduction = False

# Initialize FastAPI application
app = FastAPI()

# Configure CORS to allow all origins, methods, and headers
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

# Retrieve API keys from environment variables
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PERPLEXITY_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Note: Same as DEEPSEEK_API_KEY in original
GROK2_API_KEY = os.getenv("GROK2_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

# Global in-memory cache for search results
search_cache = {}  # Format: {query: (timestamp, result)}

# Pydantic model for chat request payload
class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str

# Asynchronous function to fetch SerpAPI search results with caching
async def get_serpapi_search_results(query: str):
    now = time.time()
    # Check if query exists in cache and is within 5 minutes
    if query in search_cache:
        timestamp, result = search_cache[query]
        if now - timestamp < 300:  # 5-minute expiration
            logger.info(f"Cache hit for query: {query}")
            return result
    logger.info(f"Cache miss for query: {query}")
    try:
        # Define target sites for search filtering
        target_sites = [
            "site:manulife.com.hk",
            "site:aia.com.hk",
            "site:prudential.com.hk",
            "site:axa.com.hk",
            "site:sunlife.com.hk",
            "site:chubb.com",
            "site:scmp.com",
            "site:hket.com",
            "site:ft.com",
            "site:moneyhero.com.hk",
            "site:compareasia.com",
            "site:policypal.com",
            "site:ia.hk",
            "site:sfc.hk",
            "site:bloomberg.com",
            "site:forbes.com"
        ]
        site_filter = f"({' OR '.join(target_sites)})"
        full_query = f"{query} {site_filter}"
        params = {
            "engine": "google",
            "q": full_query,
            "api_key": SERPAPI_API_KEY,
            "google_domain": "google.com.hk",
            "hl": "en",
            "gl": "hk",
            "num": 5
        }
        async with httpx.AsyncClient() as client:
            response = await client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
        
        # Process search results
        search_context = []
        for result in data.get("organic_results", [])[:5]:
            context_entry = (
                f"Title: {result.get('title', 'N/A')}\n"
                f"Snippet: {result.get('snippet', 'No description available')}\n"
                f"Link: {result.get('link', '')}"
            )
            search_context.append(context_entry)
        print("search_context",search_context)
        result = "\n\n".join(search_context) if search_context else "No relevant results found"
        # Cache successful results only
        if result != "No relevant results found":
            search_cache[query] = (now, result)
        return result
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return "Search service unavailable"

# Asynchronous function to fetch Google Custom Search results (unused in active endpoints)
async def get_google_search_results(query: str):
    """Get search results using Google Custom Search JSON API"""
    target_sites = [
        "site:manulife.com.hk",
        "site:aia.com.hk",
        "site:prudential.com.hk",
        "site:axa.com.hk",
        "site:sunlife.com.hk",
        "site:chubb.com",
        "site:scmp.com",
        "site:hket.com",
        "site:ft.com",
        "site:moneyhero.com.hk",
        "site:compareasia.com",
        "site:policypal.com",
        "site:ia.hk",
        "site:sfc.hk",
        "site:bloomberg.com",
        "site:forbes.com"
    ]
    site_filter = f"({' OR '.join(target_sites)})"
    full_query = f"{query} {site_filter}"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": full_query,
        "num": 5,
        "gl": "hk",
        "sort": "date",
        "fields": "items(title,link,snippet)"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            response.raise_for_status()
            data = response.json()
        
        search_context = []
        for result in data.get("items", [])[:5]:
            context_entry = (
                f"Title: {result.get('title', 'N/A')}\n"
                f"Snippet: {result.get('snippet', 'No description available')}\n"
                f"Link: {result.get('link', '')}"
            )
            search_context.append(context_entry)
        
        return "\n\n".join(search_context) if search_context else "No relevant results found"
    except Exception as e:
        logger.error(f"Google search failed: {str(e)}")
        return "Search service unavailable"

# Perplexity endpoint
@app.post("/api/ppxty")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        messages = chat_request.messages
        model = chat_request.model
        logger.info(f"Received request with messages: {messages}")
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=30.0)
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

# DeepSeek endpoint
@app.post("/api/ds")
async def deepseek_endpoint(chat_request: ChatRequest):
    try:
        messages = chat_request.messages
        model = chat_request.model
        logger.info(f"Received DeepSeek request with messages: {messages}")
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        result = response.choices[0].message.content
        logger.info(f"DeepSeek API Response: {result}")
        return {"message": result}
    except Exception as e:
        logger.error(f"Unexpected error in DeepSeek endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# DeepSeek with search endpoint
@app.post("/api/dswithsearch")
async def deepseek_with_search_endpoint(chat_request: ChatRequest):
    # Extract the latest user message
    latest_user_message = next(
        (msg for msg in reversed(chat_request.messages) if msg.get("role") == "user"),
        None
    )
    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Fetch search context with caching
    search_context = await get_serpapi_search_results(latest_user_message.get("content", ""))
    if IsProduction:
        logger.info(f"search_context={search_context}")
    else:
        print(f"search_context={search_context}")
    
    # Prepare messages with search context if available
    messages = []
    if search_context:
        messages.append({
            "role": "system",
            "content": f"Current web search context:\n{search_context}\n\nUse this information to supplement your response."
        })
    messages.extend(chat_request.messages)
    if IsProduction:
        logger.info(f"messages={messages}")
    else:
        print(f"messages={messages}")
    
    print("messages",messages)
    # Call DeepSeek API
    try:
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model=chat_request.model,
            messages=messages,
            stream=False
        )
        return {"message": response.choices[0].message.content}
    except Exception as e:
        logger.error(f"DeepSeek error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI service error")

# Startup event to initiate periodic cache cleanup
@app.on_event("startup")
async def startup_event():
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            now = time.time()
            # Remove entries older than 10 minutes
            keys_to_delete = [k for k, (t, _) in search_cache.items() if now - t > 600]
            for k in keys_to_delete:
                del search_cache[k]
    asyncio.create_task(periodic_cleanup())
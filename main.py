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

load_dotenv()

IsProduction = False

app = FastAPI()

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
    # format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PERPLEXITY_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROK2_API_KEY = os.getenv("GROK2_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")


# # PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
# PERPLEXITY_API_KEY = "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"
# if not PERPLEXITY_API_KEY:
#     raise ValueError("Missing PERPLEXITY_API_KEY environment variable")

# DEEPSEEK_API_KEY = "sk-a96c8196a00241ee9f587cf1d1f1b99d"  # Consider moving to environment variable
# if not DEEPSEEK_API_KEY:
#     raise ValueError("Missing DEEPSEEK_API_KEY environment variable")

# GOOGLE_API_KEY = "AIzaSyCihsIc9SAbQApcGcZlhwcsobzNNoDtz-s"
# GOOGLE_CX = "9280abb2866c5441d"

# GROK2_API_KEY = "xai-0fuJpGFlVbLHwO9Hi2p0uf5UWvTvViEYamWbBNpO0b78BxgKpngpmytYvdjH88ZpjOCULYpCy2fRFjSm"

class ChatRequest(BaseModel):
    messages: List[Dict]
    model: str
    
async def get_serpapi_search_results(query: str):
    """Get search results from specified domains using SerpAPI"""
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
    
    try:
        # Construct search query
        site_filter = f"({' OR '.join(target_sites)})"
        full_query = f"{query} {site_filter}"
        
        # SerpAPI parameters
        params = {
            "engine": "google",
            "q": full_query,
            "api_key": SERPAPI_API_KEY,
            "google_domain": "google.com.hk",
            "hl": "en",  # English results
            "gl": "hk",  # Hong Kong region
            "num": 5      # Number of results
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()
        
        # Process results
        search_context = []
        for result in data.get("organic_results", [])[:5]:  # Top 5 results
            context_entry = (
                f"Title: {result.get('title', 'N/A')}\n"
                f"Snippet: {result.get('snippet', 'No description available')}\n"
                f"Link: {result.get('link', '')}"
            )
            search_context.append(context_entry)
        
        
        return "\n\n".join(search_context) if search_context else "No relevant results found"
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return "Search service unavailable"

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
    
    
        
    try:
        site_filter = f"({' OR '.join(target_sites)})"
        full_query = f"{query} {site_filter}"
        
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": full_query,
            "num": 5,  # Get up to 5 results
            "gl": "hk",  # Hong Kong region
            "sort": "date",  # Sort by freshness
            "fields": "items(title,link,snippet)"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            response.raise_for_status()
            data = response.json()
        
        search_context = []
        for result in data.get("items", [])[:5]:  # Limit to 5 results
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

@app.post("/api/ppxty")
# async def chat_endpoint(request: Request, messages: list[dict]):
async def chat_endpoint(chat_request: ChatRequest):    
    try:
        messages = chat_request.messages
        model = chat_request.model
        logger.info(f"Received request with messages: {messages}")
        
        url = "https://api.perplexity.ai/chat/completions"
        
        payload = {
            # "model": "r1-1776",
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
        }
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"API Response: {result}")
            
            original_content = result['choices'][0]['message']['content']
            modified_content = original_content.replace("<think>", "<AIM AI助手>").replace("</think>", "</AIM AI助手>")
            
            return {"message": modified_content}
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# DeepSeek endpoint with asynchronous OpenAI client
@app.post("/api/ds")
# async def deepseek_endpoint(request: Request, messages: list[dict]):
async def deepseek_endpoint(chat_request: ChatRequest):
    try:
        messages = chat_request.messages
        model = chat_request.model
        logger.info(f"Received DeepSeek request with messages: {messages}")
        
        client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        response = await client.chat.completions.create(
            # model="deepseek-reasoner",
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


@app.post("/api/dswithsearch")
async def deepseek_endpoint(chat_request: ChatRequest):
    # Get latest user message
    latest_user_message = next(
        (msg for msg in reversed(chat_request.messages) if msg.get("role") == "user"),
        None
    )
    
    if not latest_user_message:
        raise HTTPException(status_code=400, detail="No user message found")
    
    # Get search context
    search_context = await get_serpapi_search_results(latest_user_message.get("content", ""))
    
    
    if IsProduction:
        logger.info("search_context=",search_context)
    else:
        print("search_context=",search_context)
            
    # Prepare messages array
    messages = []
    if search_context:
        messages.append({
            "role": "system",
            "content": f"Current web search context:\n{search_context}\n\nUse this information to supplement your response."
        })
    
    messages.extend(chat_request.messages)
    
    if IsProduction:
        logger.info("messages=",messages)
    else:
        print("messages=",messages)

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

    # # Call DeepSeek API
    # headers = {
    #     "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    #     "Content-Type": "application/json"
    # }
    
    # payload = {
    #     "messages": messages,
    #     "model": chat_request.model,
    #     # "temperature": 0.7,
    #     "max_tokens": 2000
    # }

    # try:
    #     async with httpx.AsyncClient() as client:
    #         response = await client.post(
    #             "https://api.deepseek.com/v1/chat/completions",
    #             headers=headers,
    #             json=payload
    #         )
    #         response.raise_for_status()
    #         return response.json()
    # except httpx.HTTPStatusError as e:
    #     raise HTTPException(status_code=e.response.status_code, detail=str(e))
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    
    
    
    
    
# lastone
# @app.post("/api/dswithsearch")
# # async def deepseek_endpoint(request: Request, messages: list[dict]):
# async def deepseek_endpoint(chat_request: ChatRequest):
#     try:
#         print("use deep searchxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
#          # 预定义权威网站列表
#         TRUSTED_INSURANCE_SITES = [
#             # 保险公司
#             "site:manulife.com.hk",
#             "site:aia.com.hk",
#             "site:prudential.com.hk",
#             "site:axa.com.hk",
#             "site:sunlife.com.hk",
#             "site:chubb.com",
#             # 金融媒体
#             "site:scmp.com",
#             "site:hket.com",
#             "site:ft.com",
#             # 比价平台
#             "site:moneyhero.com.hk",
#             "site:compareasia.com",
#             "site:policypal.com",
#             # 监管机构
#             "site:ia.hk",
#             "site:sfc.hk",
#             # 专业分析
#             "site:bloomberg.com",
#             "site:forbes.com"
#         ]
#         site_filters = f"({' OR '.join(TRUSTED_INSURANCE_SITES[:10])})"  # 取前10个避免超限
        
#         # 验证环境变量
#         messages = chat_request.messages
#         model = chat_request.model
#         if not all([DEEPSEEK_API_KEY, GOOGLE_API_KEY, GOOGLE_CX]):
#             raise RuntimeError("Missing API credentials in environment variables")
#         logger.info(f"Received DeepSeek request with messages: {messages}")
#         # 提取搜索查询
#         search_query = next(
#             (msg["content"] for msg in reversed(messages) 
#             if msg["role"] == "user"
#         ), None)
#         search_results = []
#         if search_query:
#             try:
#                 async with httpx.AsyncClient() as client:
#                     # 先验证API连通性
#                     # test_params = {
#                     #     "key": GOOGLE_API_KEY,
#                     #     "cx": GOOGLE_CX,
#                     #     "q": "API connectivity test",
#                     #     "num": 1
#                     # }
#                     # test_response = await client.get(
#                     #     "https://www.googleapis.com/customsearch/v1",
#                     #     params=test_params
#                     # )
#                     # test_response.raise_for_status()
#                     # 执行实际搜索
#                     enhanced_query = f"{search_query} {site_filters}"
#                     search_params = {
#                         "key": GOOGLE_API_KEY,
#                         "cx": GOOGLE_CX,
#                         "q": enhanced_query,
#                         "num": 10,
#                         "hl": "zh-CN",
#                         "sort": "date",  # 优先最新内容
#                         "cr": "countryHK",  # 限定香港地区
#                         "gl": "hk"  # 香港谷歌版本
#                     }
#                     response = await client.get(
#                         "https://www.googleapis.com/customsearch/v1",
#                         params=search_params
#                     )
#                     response.raise_for_status()
                    
#                     data = response.json()
#                     search_results = data.get("items", [])
#                     logger.info(f"Google search returned {len(search_results)} results")
#             except httpx.HTTPStatusError as e:
#                 logger.error(f"Google API error: {e.response.text}")
#                 raise HTTPException(
#                     status_code=502,
#                     detail=f"Search service error: {e.response.text}"
#                 )
#             except Exception as e:
#                 logger.warning(f"Google search failed: {str(e)}")
#                 search_results = []
#         # Build search context if results found
#         if search_results:
#             search_context = "Latest web search results:\n"
#             for idx, item in enumerate(search_results[:10], 1):  # Use top 3 results
#                 search_context += (
#                     f"{idx}. [{item.get('title', 'No title')}]({item.get('link', '')})\n"
#                     f"{item.get('snippet', 'No description available')}\n\n"
#                 )
            
#             # Insert search context before the last user message
#             for idx in reversed(range(len(messages))):
#                 if messages[idx]["role"] == "user":
#                     messages.insert(idx, {"role": "system", "content": search_context})
#                     break
#         # Call DeepSeek API
#         print("messages",messages)
#         client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
#         response = await client.chat.completions.create(
#             # model="deepseek-reasoner",
#             model=model,
#             messages=messages,
#             stream=False
#         )
#         result = response.choices[0].message.content
#         logger.info(f"DeepSeek API response generated")
        
#         return {"message": result}
#     except Exception as e:
#         logger.error(f"Endpoint error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))  


# @app.post("/api/grok2")
# # async def deepseek_endpoint(request: Request, messages: list[dict]):
# async def grok2_endpoint(chat_request: ChatRequest):
#     try:
#         messages = chat_request.messages
#         model = chat_request.model
#         logger.info(f"Received Grok2 request with messages: {messages}")
        
#         client = AsyncOpenAI(api_key=GROK2_API_KEY, base_url="https://api.x.ai/v1")
        
#         response = await client.chat.completions.create(
#             # model="deepseek-reasoner",
#             model=model,
#             messages=messages,
#             # stream=False
#         )
        
#         result = response.choices[0].message.content
#         logger.info(f"Grok2 API Response: {result}")
        
#         return {"message": result}
        
#     except Exception as e:
#         logger.error(f"Unexpected error in Grok endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))        
    
# @app.post("/api/ppxty")
# async def chat_endpoint(request: Request, messages: list[dict]):
    
#     try:
#         logger.info(f"Received request with messages: {messages}")
        
#         url = "https://api.perplexity.ai/chat/completions"
        
#         # Add system message with instructions
#         # system_message = {
#         #     "role": "system",
#         #     "content": "用超級詳盡的方式比較, 例如要有基礎保單架構,核心保障差異,所有比較都是用表格形式顯示"
#         # }
        
#         # Create modified messages array with system message first
#         # modified_messages = [system_message] + messages
        
#         payload = {
#             # "model": "sonar-deep-research",
#             "model": "r1-1776",
#             "messages": messages,  # Use modified messages
#             "max_tokens": 2000,
#         }
        
#         headers = {
#             "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
#             "Content-Type": "application/json"
#         }
        
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()
        
#         result = response.json()
#         logger.info(f"API Response: {result}")
        
#         # Replace the tags in the response
#         original_content = result['choices'][0]['message']['content']
#         modified_content = original_content.replace("<think>", "<AIM AI助手>").replace("</think>", "</AIM AI助手>")
        
#         return {"message": modified_content}
        
#     except requests.exceptions.HTTPError as e:
#         logger.error(f"HTTP Error: {e.response.text}")
#         raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))   
    
# @app.post("/api/ds")
# async def deepseek_endpoint(request: Request, messages: list[dict]):
    
#     try:
#         logger.info(f"Received DeepSeek request with messages: {messages}")
        
#         client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
#         # Add system message if not already present
#         # if messages and messages[0]["role"] != "system":
#         #     system_message = {"role": "system", "content": "用超級詳盡的方式比較, 例如要有基礎保單架構,核心保障差異,所有比較都是用表格形式顯示"}
#         #     modified_messages = [system_message] + messages
#         # else:
#         #     modified_messages = messages
        
#         response = client.chat.completions.create(
#             model="deepseek-reasoner",
#             # model="deepseek-chat",
#             messages=messages,
#             stream=False
#         )
        
#         result = response.choices[0].message.content
#         logger.info(f"DeepSeek API Response: {result}")
        
#         return {"message": result}
        
#     except Exception as e:
#         logger.error(f"Unexpected error in DeepSeek endpoint: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
         
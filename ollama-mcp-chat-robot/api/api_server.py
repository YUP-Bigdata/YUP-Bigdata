from fastapi import FastAPI, Request
from pydantic import BaseModel
import asyncio
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import logging
from fastapi.middleware.cors import CORSMiddleware

from agent.llm_ollama import OllamaAgentManager
from mcp_server.mcp_manager import MCPManager
from app_settings import AppSettings
from agent.chat_storage import ChatStorage
from constants import DEFAULT_SYSTEM_PROMPT


# from constants import MISTRAL_API

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 "http://localhost:5500" 등으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 변수 초기화
agent_manager = None
chat_storage = None
loop = asyncio.get_event_loop()

class Message(BaseModel):
    role: str  # "user" 또는 "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    image_url: Optional[str] = None
    new_chat: Optional[bool] = False
    explicit_command: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    function: Optional[str] = None

@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 상태 확인용
    """
    return {
        "status": "ok",
        "message": "Robot Command API Server is running"
    }

mcp_client = None
loop = asyncio.get_event_loop()

@app.on_event("startup")
async def startup_event():
    global agent_manager, chat_storage, mcp_client  # chat_history 대신 chat_storage
    config = AppSettings().getAll()
    temperature = config.get("temperature", {}).get("value", 0.1)

    try:
        print("\n=== Initializing MCP client... ===")
        # MCP 설정 로드
        mcp_config = MCPManager().getConfig()
        
        # MCP 클라이언트 초기화
        from langchain_mcp_adapters.client import MultiServerMCPClient
        mcp_client = MultiServerMCPClient(mcp_config["mcpServers"])
        
        # worker.py와 동일한 초기화 로직 적용
        await asyncio.sleep(1)  # 클라이언트 초기화 대기
        await mcp_client.__aenter__()
        await asyncio.sleep(1)  # 연결 안정화 대기
        
        # MCP 도구 가져오기
        tools = mcp_client.get_tools()
        print(f"Loaded {len(tools)} MCP tools.")
        for tool in tools:
            print(f"[Tool] {tool.name}")
        
        # Agent 초기화 (worker.py와 동일한 방식)
        agent_manager = OllamaAgentManager(
            mcp_client=mcp_client,  
            mcp_tools=tools         
        )
        # agent_manager = MistralAgentManager(
        #     api_key=MISTRAL_API,    
        #     mcp_client=mcp_client,  
        #     mcp_tools=tools         
        # )
        agent_manager.createChatModel(
            temperature=temperature,
            mcp_tools=tools,
        )
        chat_storage = ChatStorage()
        await chat_storage.init()
        
        print("=== MCP initialization complete ===\n")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        if mcp_client:
            await mcp_client.__aexit__(None, None, None)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global agent_manager, mcp_client
    try:
        # MCP 클라이언트 정리 (worker.py의 cleanup 로직 적용)
        if mcp_client:
            await mcp_client.__aexit__(None, None, None)
            print("MCP client closed")
        if agent_manager and agent_manager.mcp_client:
            await agent_manager.mcp_client.__aexit__(None, None, None)
            print("Agent MCP client closed")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)

@app.get("/chats")
async def get_chat_list():
    """대화 목록 조회"""
    global chat_storage
    chats = await chat_storage.list_chats()
    return {"chats": chats}

@app.get("/chats/{chat_id}")
async def get_chat_messages(chat_id: str):
    """특정 대화의 메시지 조회"""
    global chat_storage
    try:
        chat = await chat_storage.get_chat(chat_id)
        return {"messages": chat["messages"]}
    except Exception:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=404,
            content={"error": "Chat not found"}
        )

Prompt = """
You are an expert robotic task planner with strong visual reasoning skills.

Your goal is to analyze the given **explicit command** and **attached image**, and generate a step-by-step robot action plan that can be executed in order.

---

## Input Context

- An image is provided.
- Explicit high-level command: {explicit_command}
- Target object attributes:
  - Color: {Color}
  - Size: {Size}
  - Function: {Function}

---

## Output Requirements

1. Carefully analyze the image to identify:
   - Relevant objects and their spatial positions
   - Colors, sizes, functions, and environmental constraints
   - **Precise rack layout and object placement (e.g., which row and column each item occupies)**

2. Decompose the explicit command into a **step-by-step plan** for the robot to execute.

3. Each step must be:
   - Concise and directly executable
   - Ordered logically (from step1 to stepN)
   - Informed by the image and explicit command

4. Do not include explanations or rationales—only the action instructions.

5. Format the output in the following JSON structure:

```json
{{
  "image_description": "Detailed and factual description of the scene, including the positions of all visible containers within the racks (e.g., 'a small yellow and white pill container is inserted in row 1, column 3 of the right rack').",
  "action_plan": {{
    "step1": "First robot action here.",
    "step2": "Second robot action here.",
    "step3": "Third robot action here"
    // Add more steps as needed.
  }}
}}
"""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    global agent_manager, chat_storage
    config = AppSettings().getAll()
    timeout = config.get("timeout", {}).get("value", 600)
    
    start_time = time.time()
    
    try:
        if "Explicit command:" not in req.query:
            raise ValueError("Invalid query format: Explicit command not found")

        # 예: "/home/xxx.jpg Explicit command: ..."
        parts = req.query.split("Explicit command:")
        explicit_part = parts[1].strip()  # "Pick up ... Color : xxx, Size : xxx, Function : xxx"

        command_and_attrs = explicit_part.split("Color :")
        explicit_command = command_and_attrs[0].strip()

        attrs = command_and_attrs[1].split(", Size :")
        color = attrs[0].strip()

        size_and_function = attrs[1].split(", Function :")
        size = size_and_function[0].strip()
        function = size_and_function[1].strip()

        
        formatted_prompt = Prompt.format(
            explicit_command=explicit_command,
            Color=color,
            Size=size,
            Function=function
        )

        # --- LLM 호출 ---
        result = await agent_manager.chat(
            # query= req.query,
            query=DEFAULT_SYSTEM_PROMPT + "\n" + formatted_prompt,
            timeout=timeout
        )

        if "output" not in result:
            return JSONResponse(
                status_code=500,
                content={"error": "LLM 응답 생성 실패"}
            )

        # --- 채팅 관리 ---
        new_messages = [
            f"User: {req.query}",
            f"Assistant: {result['output']}"
        ]

        if req.new_chat or req.chat_id is None:
            chat_id = await chat_storage.create_chat(
                # title=explicit_command[:10] + "...",
                title=req.query[:10] + "...",
                messages=new_messages
            )
            updated_history = new_messages
        else:
            chat = await chat_storage.get_chat(req.chat_id)
            updated_history = chat["messages"] + new_messages
            await chat_storage.update_chat(req.chat_id, updated_history)
            chat_id = req.chat_id

        end_time = time.time()

        return {
            "chat_id": chat_id,
            "output": result["output"],
            "processing_time_seconds": round(end_time - start_time, 2)
        }

    except Exception as e:
        end_time = time.time()
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"오류 발생: {str(e)}",
                "processing_time_seconds": round(end_time - start_time, 2)
            }
        )

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """채팅 삭제"""
    global chat_storage
    try:
        await chat_storage.delete_chat(chat_id)
        return {"status": "success"}
    except Exception:
        return JSONResponse(
            status_code=404,
            content={"error": "Chat not found"}
        )
    
@app.on_event("shutdown")
async def shutdown_event():
    global agent_manager, mcp_client, chat_storage
    try:
        if mcp_client:
            await mcp_client.__aexit__(None, None, None)
            print("MCP client closed")
        if agent_manager and agent_manager.mcp_client:
            await agent_manager.mcp_client.__aexit__(None, None, None)
            print("Agent MCP client closed")
        if chat_storage:
            await chat_storage.close()
            print("Chat storage closed")
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
import asyncio
import uuid
from queue import Queue
from typing import Any, List, Optional
import os
import base64
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_QUERY_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    RECURSION_LIMIT,
)


class OllamaAgentManager:
    QUERY_THREAD_ID = str(uuid.uuid4())
    USE_ASTREAM_LOG = True
    QWEN3 = DEFAULT_LLM_MODEL
    print(QWEN3)

    def __init__(
        self,
        mcp_client: Optional[MultiServerMCPClient] = None,
        mcp_tools: Optional[List] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        out_queue: Optional[Queue] = None,
    ):
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.agent = None
        self.mcp_client = mcp_client
        self.mcp_tools = mcp_tools
        self.is_first_chat = True
        self.out_queue = out_queue

    def createChatModel(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        mcp_tools: Optional[List] = None,
    ) -> ChatOllama | CompiledGraph:
        self.agent = ChatOllama(
            model=OllamaAgentManager.QWEN3,
            temperature=temperature,
        )
        if mcp_tools:
            self.agent = create_react_agent(
                model=self.agent, tools=mcp_tools, checkpointer=MemorySaver()
            )
            print("ReAct agent created.")
            return self.agent
        else:
            print("No MCP tools provided. Using plain Gemini model.")
            return self.agent

    def getStreamingCallback(self):
        accumulated_text = []
        accumulated_tool_info = []

        def callback_func(data: Any):
            nonlocal accumulated_text, accumulated_tool_info

            if isinstance(data, dict):
                agent_step_key = next(
                    (
                        k
                        for k in data
                        if isinstance(data.get(k), dict) and "messages" in data[k]
                    ),
                    None,
                )

                if agent_step_key:
                    messages = data[agent_step_key].get("messages", [])
                    for message in messages:
                        if isinstance(message, AIMessage):
                            if message.tool_calls:
                                pass
                            elif message.content and isinstance(message.content, str):
                                content_chunk = message.content.encode(
                                    "utf-8", "replace"
                                ).decode("utf-8")
                                if content_chunk:
                                    accumulated_text.append(content_chunk)
                                    print(content_chunk, end="", flush=True)
                                    if self.out_queue:
                                        self.out_queue.put(
                                            {
                                                "type": "chat_message",
                                                "data": content_chunk,
                                            }
                                        )

                        elif isinstance(message, ToolMessage):
                            tool_info = f"Tool Used: {message.name}\nResult: {message.content}\n---------------------"
                            print(f"\n[Tool Execution Result: {message.name}]")
                            accumulated_tool_info.append(tool_info)
                            if self.out_queue:
                                self.out_queue.put(
                                    {"type": "chat_message", "data": tool_info}
                                )
            return None

        return callback_func, accumulated_text, accumulated_tool_info

    async def processQuery(
        self,
        agent,
        system_prompt: Optional[str],
        query: str,
        timeout: int = DEFAULT_QUERY_TIMEOUT,
    ):
        try:
            streaming_callback, accumulated_text, accumulated_tool_info = self.getStreamingCallback()

            # ✅ Local image path detection & base64 encoding
            image_pattern = r'(/[^\s]+\.(jpg|jpeg|png|gif|bmp|png))'
            image_paths = re.findall(image_pattern, query)

            if image_paths:
                text_only = re.sub(image_pattern, '', query).strip()
                messages_content = []
                if text_only:
                    messages_content.append({"type": "text", "text": text_only})

                for path, ext in image_paths:
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            encoded = base64.b64encode(f.read()).decode("utf-8")
                            data_url = f"data:image/{ext.lower()};base64,{encoded}"
                            messages_content.append({
                                "type": "image_url",
                                "image_url": data_url
                            })
                        print(f"[OK] Encoded local image: {path}")
                    else:
                        print(f"[WARN] Skipped non-existing image path: {path}")
            else:
                messages_content = query

            # ✅ Build messages
            if system_prompt:
                initial_messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=messages_content)
                ]
            else:
                initial_messages = [HumanMessage(content=messages_content)]

            inputs = {"messages": initial_messages}
            config = RunnableConfig(
                recursion_limit=RECURSION_LIMIT,
                configurable={"thread_id": self.QUERY_THREAD_ID},
            )

            if self.USE_ASTREAM_LOG:
                async for chunk in agent.astream_log(
                    inputs, config=config, include_types=["llm", "tool"]
                ):
                    streaming_callback(chunk.ops[0]["value"])
            else:
                response = await agent.ainvoke(inputs, config=config)
                if isinstance(response, dict) and "messages" in response:
                    final_message = response["messages"][-1]
                    if isinstance(final_message, (AIMessage, ToolMessage)):
                        content = final_message.content
                        print(content, end="", flush=True)
                        accumulated_text.append(content)

                        if isinstance(final_message, AIMessage) and final_message.additional_kwargs.get("tool_calls"):
                            for tool_call in final_message.additional_kwargs["tool_calls"]:
                                tool_info = f"Tool Used: {tool_call.get('name', 'Unknown')}\n"
                                accumulated_tool_info.append(tool_info)

            return {
                "output": "".join(accumulated_text).strip(),
                "tool_calls": "\n".join(accumulated_tool_info)
            }

        except asyncio.TimeoutError:
            return {"error": f"⏱️ Request exceeded timeout of {timeout} seconds."}
        except Exception as e:
            import traceback
            print(f"\nDebug info: {traceback.format_exc()}")
            return {"error": f"❌ An error occurred: {str(e)}"}


    async def chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        timeout: int = DEFAULT_QUERY_TIMEOUT,
        out_queue: Optional[Queue] = None,
    ):
        if self.agent is None:
            raise RuntimeError("Agent is not initialized. Call initialize() first.")
        if self.is_first_chat:
            system_prompt = system_prompt or self.system_prompt
            self.is_first_chat = False
        else:
            system_prompt = ""
        return await self.processQuery(
            self.agent,
            system_prompt,
            query,
            timeout=timeout or DEFAULT_QUERY_TIMEOUT,
        )

    def reset(
        self, temperature: float = DEFAULT_TEMPERATURE, mcp_tools: Optional[List] = None
    ):
        """
        대화(히스토리)를 완전히 초기화합니다.
        temperature, mcp_tools를 지정하지 않으면 기존 값 사용.
        """
        tools = mcp_tools if mcp_tools is not None else self.mcp_tools
        self.createChatModel(temperature=temperature, mcp_tools=tools)
        self.is_first_chat = True
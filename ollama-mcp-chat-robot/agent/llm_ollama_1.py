import asyncio
import uuid
import re
import os
import base64
from queue import Queue
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mistralai import ChatMistralAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_QUERY_TIMEOUT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    RECURSION_LIMIT,
)


class MistralAgentManager:
    QUERY_THREAD_ID = str(uuid.uuid4())
    USE_ASTREAM_LOG = True

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        mcp_client: Optional[MultiServerMCPClient] = None,
        mcp_tools: Optional[List] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        out_queue: Optional[Queue] = None,
    ):
        self.api_key = api_key
        self.model = model
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
    ):
        mistral_llm = ChatMistralAI(
            api_key=self.api_key,
            model=self.model,
            temperature=temperature,
        )

        if mcp_tools:
            self.agent = create_react_agent(
                model=mistral_llm,
                tools=mcp_tools,
                checkpointer=MemorySaver()
            )
            print("✅ ReAct Agent created with ChatMistralAI backend.")
        else:
            self.agent = mistral_llm
            print("✅ Plain ChatMistralAI model created.")

        return self.agent

    def getStreamingCallback(self):
        accumulated_text = []
        accumulated_tool_info = []

        def callback_func(data: Any):
            nonlocal accumulated_text, accumulated_tool_info
            if isinstance(data, dict):
                agent_step_key = next(
                    (k for k in data if isinstance(data[k], dict) and "messages" in data[k]),
                    None,
                )
                if agent_step_key:
                    messages = data[agent_step_key].get("messages", [])
                    for message in messages:
                        if isinstance(message, AIMessage):
                            if message.tool_calls:
                                pass
                            elif message.content and isinstance(message.content, str):
                                chunk = message.content.strip()
                                if chunk:
                                    accumulated_text.append(chunk)
                                    print(chunk, end="", flush=True)
                                    if self.out_queue:
                                        self.out_queue.put({
                                            "type": "chat_message",
                                            "data": chunk
                                        })
                        elif isinstance(message, ToolMessage):
                            tool_info = f"[Tool: {message.name}] {message.content}"
                            print(tool_info)
                            accumulated_tool_info.append(tool_info)
                            if self.out_queue:
                                self.out_queue.put({
                                    "type": "chat_message",
                                    "data": tool_info
                                })
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

    async def chat(self, query: str, chat_history: Optional[List[str]] = None, timeout: int = 600):
        system_prompt = None
        if chat_history:
            system_prompt = "Previous chat history:\n" + "\n".join(chat_history)

        result = await self.processQuery(
            agent=self.agent,
            system_prompt=system_prompt,
            query=query,
            timeout=timeout
        )
        if "error" in result:
            raise Exception(result["error"])
        return {
            "output": result["output"],
            "tool_calls": result.get("tool_calls", "")
        }

    def reset(self, temperature: float = DEFAULT_TEMPERATURE, mcp_tools: Optional[List] = None):
        tools = mcp_tools if mcp_tools is not None else self.mcp_tools
        self.createChatModel(temperature=temperature, mcp_tools=tools)
        self.is_first_chat = True

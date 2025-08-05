import asyncio
import queue

from langchain_mcp_adapters.client import MultiServerMCPClient
from PySide6.QtCore import QObject, Signal

from agent.llm_ollama import OllamaAgentManager
from app_settings import AppSettings
from constants import EVENT_DATA, EVENT_TYPE
from mcp_server.mcp_manager import MCPManager


class Worker(QObject):
    finished = Signal(object)

    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.running = True
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.mcp_client = None
        self.mcp_tools = None
        self.agent_manager = None
        self.mcp_manager = MCPManager()

    def run(self):
        while self.running:
            try:
                event = self.in_queue.get(timeout=0.1)
                # event = {EVENT_TYPE: str, EVENT_DATA: Any}
                if event[EVENT_TYPE] == "init":
                    self.loop.run_until_complete(self.initializeMCP())
                elif event[EVENT_TYPE] == "reset_chat":
                    self.loadAppSettings()
                    if self.agent_manager:
                        self.agent_manager.reset(
                            temperature=self.temperature, mcp_tools=self.mcp_tools
                        )
                elif event[EVENT_TYPE] == "chat":
                    if self.agent_manager:
                        self.loadAppSettings()
                        result = self.loop.run_until_complete(
                            self.agent_manager.chat(
                                event[EVENT_DATA],
                                system_prompt=self.system_prompt,
                                timeout=self.timeout,
                                out_queue=self.out_queue,
                            )
                        )
                        self.out_queue.put(
                            {EVENT_TYPE: "chat_result", EVENT_DATA: result}
                        )
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        if self.mcp_client:
            self.loop.run_until_complete(self.cleanupMCPClient(self.mcp_client))
        self.loop.close()

    def loadAppSettings(self):
        self.config = AppSettings().getAll()
        self.temperature = self.config.get("temperature", {}).get("value", 1)
        self.system_prompt = self.config.get("system_prompt", {}).get("value", "")
        self.timeout = self.config.get("timeout", {}).get("value", 5 * 60)

    async def initializeMCP(self):
        self.loadAppSettings()

        print("\n=== Initializing MCP client... ===")
        mcp_config = self.mcp_manager.getConfig()
        self.mcp_client = MultiServerMCPClient(mcp_config["mcpServers"])
        await asyncio.sleep(1)
        await self.mcp_client.__aenter__()
        await asyncio.sleep(1)

        self.mcp_tools = self.mcp_client.get_tools()
        print(f"Loaded {len(self.mcp_tools)} MCP tools.")
        for tool in self.mcp_tools:
            print(f"[Tool] {tool.name}")

        self.agent_manager = OllamaAgentManager(self.mcp_client, self.mcp_tools)
        self.agent_manager.createChatModel(
            temperature=self.temperature,
            mcp_tools=self.mcp_tools,
        )
        self.out_queue.put({"type": "init_done"})

    async def cleanupMCPClient(self, client=None):
        if client is not None:
            await client.__aexit__(None, None, None)

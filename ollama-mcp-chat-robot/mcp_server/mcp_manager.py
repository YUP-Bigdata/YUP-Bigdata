import json
import os

DEFAULT_MCP_CONFIG = {"mcpServers": {}}
MCP_ADD_TEMPLATES = {
    "command": "mcp_server_command",
    "args": ["mcp_server_arg1", "mcp_server_arg2"],
    "transport": "set_stdio_for_local_server",
}
MCP_CONFIG_PATH = "mcp_config.json"


class MCPManager:
    def __init__(self):
        self.config = self.loadConfigFile()

    def loadConfigFile(self):
        if not os.path.exists(MCP_CONFIG_PATH):
            return {"mcpServers": {}}
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def getConfig(self, key=None):
        if key is None:
            return self.config
        else:
            return self.config[key]

    def saveConfig(self, config):
        with open(MCP_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def validateConfig(self, config):
        # check if "mcpServers" is in the top level of the config
        if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
            return (
                False,
                '"mcpServers" key is not in the top level of the config or is not a dictionary.',
            )
        # check if each server object has the required parameters
        for name, server in config["mcpServers"].items():
            if not isinstance(server, dict):
                return False, f'"{name}" server config is not a dictionary.'
            if "command" not in server or "args" not in server:
                return False, f'"{name}" server has no "command" or "args".'
            if not isinstance(server["args"], list):
                return False, f'"{name}" server "args" must be a list.'
        return True, "Valid MCP config."

    def addServer(self, name, command, args, extra_params=None):
        config = self.loadConfigFile()
        if name in config["mcpServers"]:
            raise ValueError(f"{name} server already exists.")
        server_obj = {"command": command, "args": args}
        if extra_params:
            server_obj.update(extra_params)
        config["mcpServers"][name] = server_obj
        valid, msg = self.validateConfig(config)
        if not valid:
            raise ValueError(f"Configuration error: {msg}")
        self.saveConfig(config)
        print(f"{name} server added successfully.")

    @staticmethod
    async def testMCPTool(mcp_tools):
        try:
            for tool in mcp_tools:
                print(f"[Tool] {tool.name}")
                # print(f"  - Description: {tool.description}")
                # print(f"  - Schema: {tool.args_schema}")
                # if tool.name == "get_weather":
                #     try:
                #         test_response = await tool.ainvoke({"location": "Seoul"})
                #         print("\n=== MCP Server Status ===")
                #         print("✅ MCP server is running normally.")
                #         print(f"Test response: {test_response}")
                #     except Exception as e:
                #         print("\n❌ MCP server status check failed:")
                #         print(f"- Error message: {str(e)}")
                #         return
        except Exception as e:
            print(f"Error occurred during test call: {str(e)}")

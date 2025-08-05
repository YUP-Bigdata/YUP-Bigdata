EVENT_TYPE = "type"
EVENT_DATA = "data"

DEFAULT_LLM_MODEL = "llama4:17b-scout-16e-instruct-q8_0"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_QUERY_TIMEOUT = 60 * 5
RECURSION_LIMIT = 100
DEFAULT_SYSTEM_PROMPT = """
        You are a helpful AI assistant that can use tools to answer questions.
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do.
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        When using tools, think step by step:
        1. Understand the question and what information is needed.
        2. Look at the available tools ({tool_names}) and their descriptions ({tools}).
        3. Decide which tool, if any, is most appropriate to find the needed information.
        4. Determine the correct input parameters for the chosen tool based on its description.
        5. Call the tool with the determined input.
        6. Analyze the tool's output (Observation).
        7. If the answer is found, formulate the Final Answer. If not, decide if another tool call is needed or if you can answer based on the information gathered.
        8. Only provide the Final Answer once you are certain. Do not use a tool if it's not necessary to answer the question.
        """

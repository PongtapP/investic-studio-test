import logging
from typing import Dict, Any, List

# Import LangChain components for agent creation and tool binding
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, StructuredTool

# Import custom components.
from .gemini_client import GeminiClient
from .tools import Tools, InternalQAToolInput, IssueSummaryToolInput

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIAgent:
    """
    The core AI agent that uses an LLM to route queries
    to the appropriate tools.
    """

    def __init__(self, client: GeminiClient, tools: Tools):
        """
        Initializes the AI agent with a Gemini client and a set of tools.

        Args:
            client (GeminiClient): The Gemini client for LLM communication.
            tools (Tools): The Tools instance containing the available tools.
        """
        self.client = client
        self.tools = tools
        self._initialize_agent()
        logging.info("AI Agent initialization complete.")

    def _initialize_agent(self):
        """
        Initializes the LangChain agent with the LLM and tools.
        """
        logging.info("Creating agent executor...")
        
        # Get the LLM instance from the client
        llm = self.client.get_chat_model()
        
        # Manually create StructuredTool objects to ensure proper binding.
        # This resolves the `missing 'self'` error during tool execution.
        tool_list = [
            StructuredTool.from_function(
                func=self.tools.internal_qa_tool,
                name="internal_qa_tool",
                description="Tool to answer questions based on internal documents. It searches the vector store for relevant information and uses the LLM to generate an answer.",
                args_schema=InternalQAToolInput
            ),
            StructuredTool.from_function(
                func=self.tools.issue_summary_tool,
                name="issue_summary_tool",
                description="Tool to summarize a user-reported issue. It extracts key information like reported issues, affected components, and severity.",
                args_schema=IssueSummaryToolInput
            )
        ]

        # Define the system prompt for the agent
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Your primary function is to route user queries to the appropriate tool. Use the 'internal_qa_tool' for questions related to internal documents, and use the 'issue_summary_tool' for user bug reports or issues."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # Create the LangChain agent
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tool_list,
            prompt=prompt_template
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tool_list,
            verbose=True
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a user query by running it through the agent executor.

        Args:
            query (str): The user's input query.

        Returns:
            Dict[str, Any]: The agent's response, which may be a structured output
                            from a tool or a final answer from the LLM.
        """
        try:
            logging.info(f"Agent received query: {query}")
            # The agent will now pass the 'tools' instance to the tool functions
            response = self.agent_executor.invoke({"input": query})
            # The response is a dictionary, return it directly
            return response
        except Exception as e:
            logging.error(f"An error occurred while running the agent: {e}")
            return {"error": str(e)}

# Test run for internal Q&A tool
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    # Load environment variables from the parent directory
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        logging.info("Initializing AI Agent...")
        gemini_client = GeminiClient(api_key=api_key)
        tools_instance = Tools(client=gemini_client)
        agent = AIAgent(client=gemini_client, tools=tools_instance)

        logging.info("\n--- Testing with Internal Q&A Tool ---")
        qa_query = "What did users say about the search bar?"
        qa_response = agent.process_query(qa_query)
        print("Agent Response:")
        print(qa_response)

        logging.info("\n--- Testing with Issue Summary Tool ---")
        issue_query = "The 'share' button is not working when I use the app on my phone."
        issue_response = agent.process_query(issue_query)
        print("Agent Response:")
        print(issue_response)

        logging.info("\n--- Testing with an Unrelated Query ---")
        unrelated_query = "What is the capital of France?"
        unrelated_response = agent.process_query(unrelated_query)
        print("Agent Response:")
        print(unrelated_response)
        
    except ValueError as e:
        logging.error(f"An error occurred during agent testing: {e}")
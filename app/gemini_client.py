import os
import logging
from typing import Optional

# Import Google Generative AI library
import google.generativeai as genai

# Import LangChain models for compatibility
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiClient:
    """
    A client class to handle interactions with Google's Gemini models
    for both chat and embedding functionalities.
    """

    def __init__(self, api_key: str):
        """
        Initializes the GeminiClient with the provided API key.

        Args:
            api_key (str): The API key for accessing the Gemini API.
        """
        if not api_key:
            raise ValueError("API key must be provided to initialize GeminiClient.")
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        logging.info("Gemini client configured successfully.")

    def get_chat_model(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Returns a LangChain-compatible chat model instance.

        Args:
            model_name (str): The name of the Gemini chat model to use.
            temperature (float): The sampling temperature.
                                 Lower values lead to more deterministic responses.

        Returns:
            ChatGoogleGenerativeAI: An instance of the chat model.
        """
        logging.info(f"Creating chat model instance: {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=self.api_key # Pass the API key explicitly
        )

    def get_embedding_model(self, model_name: str = "models/embedding-001"):
        """
        Returns a LangChain-compatible embedding model instance.

        Args:
            model_name (str): The name of the Gemini embedding model to use.

        Returns:
            GoogleGenerativeAIEmbeddings: An instance of the embedding model.
        """
        logging.info(f"Creating embedding model instance: {model_name}")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=self.api_key # Pass the API key explicitly
        )

    def invoke_chat_model(self, prompt: str) -> str:
        """
        Invokes the chat model with a given prompt and returns the string response.
        
        Args:
            prompt (str): The input prompt for the model.
            
        Returns:
            str: The generated text response.
        """
        logging.info("Invoking Gemini chat model directly.")
        model = self.get_chat_model()
        response = model.invoke(prompt)
        return response.content

# Example usage for direct testing
if __name__ == "__main__":
    
    # Load environment variables from the parent directory
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    # Get API key from environment variable
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # Initialize the client
        client = GeminiClient(api_key=api_key)
        
        # Test chat model
        chat_model = client.get_chat_model()
        print("\nChat model instance created successfully.")

        # Test embedding model
        embedding_model = client.get_embedding_model()
        print("\nEmbedding model instance created successfully.")
        
    except ValueError as e:
        logging.error(f"Error during testing: {e}")
import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import the core components
from .gemini_client import GeminiClient
from .tools import Tools
from .agent import AIAgent

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure .env file is loaded correctly regardless of current working directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

class QueryPayload(BaseModel):
    """
    Pydantic model for the incoming JSON payload.
    """
    query: str

# Create the FastAPI application instance
app = FastAPI(
    title="INVESTIC STUDIO TEST API",
    description="An AI agent that routes queries to internal Q&A and issue summary tools.",
    version="1.0.0"
)

# Initialize the core components
@app.on_event("startup")
async def startup_event():
    """
    Initialize all components once when the application starts.
    """
    logging.info("Initializing application components...")
    try:
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        gemini_client = GeminiClient(api_key=api_key)
        tools = Tools(client=gemini_client)
        ai_agent = AIAgent(client=gemini_client, tools=tools)
        app.state.ai_agent = ai_agent
        logging.info("Application components initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize application: {e}")
        # Make the exception message more specific
        raise HTTPException(status_code=500, detail=f"Failed to start up the AI assistant: {e}")


@app.post("/process")
async def process_query(payload: QueryPayload) -> Dict[str, Any]:
    """
    API endpoint to process a user query using the AI Agent.
    """
    logging.info(f"Received new query: '{payload.query}'")
    
    if not hasattr(app.state, 'ai_agent'):
        raise HTTPException(status_code=503, detail="AI assistant is not yet ready. Please try again later.")
    
    try:
        response = app.state.ai_agent.process_query(payload.query)
        logging.info("Query processed successfully.")
        return response
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")

if __name__ == "__main__":
    # ** FIX: Ensure the server runs with the correct reload mechanism **
    # Changed from uvicorn.run(app, ...) to a more robust setup for reload.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
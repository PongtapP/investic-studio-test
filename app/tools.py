import os
import logging
import json
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InternalQAToolInput(BaseModel):
    """Input schema for the internal_qa_tool."""
    query: str = Field(..., description="The user's question to be answered.")

class IssueSummaryToolInput(BaseModel):
    """Input schema for the issue_summary_tool."""
    query: str = Field(..., description="The raw text of the user-reported issue to be summarized.")

class Tools:
    """
    A class that manages all the tools for the AI agent.
    This includes document ingestion, vector store management,
    and defining the tools themselves.
    """

    def __init__(self, client: GeminiClient):
        """
        Initializes the Tools class.
        
        Args:
            client (GeminiClient): An instance of the GeminiClient.
        """
        self.client = client
        self.vector_store = None
        self._ingest_data()
        logging.info("Tools initialized and data ingested.")

    def _ingest_data(self):
        """
        Loads documents, splits them, and creates a vector store.
        This method is called during initialization to prepare the data for the agent.
        """
        logging.info("Starting data ingestion process...")
        documents: List[Document] = []
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if file.endswith(".pdf"):
                logging.info(f"Loading PDF file: {file}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".docx"):
                logging.info(f"Loading DOCX file: {file}")
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            logging.warning("No documents found to load. Please check the 'data' directory.")
            return

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Loaded {len(documents)} documents and split them into {len(chunks)} chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=self.client.get_embedding_model()
        )
        logging.info("Vector store created successfully.")

    def _get_issue_summary_json(self, query: str, attempt: int = 1) -> Optional[Dict[str, Any]]:
        """
        Helper function to get a JSON summary from the LLM, with a retry mechanism.
        """
        try:
            prompt = f"""
            You are an expert bug report summarizer.
            Analyze the following user-reported issue and extract key information.

            User report: "{query}"

            You must respond with a single, valid JSON object containing the following keys:
            - "Reported issues": A brief summary of the problem.
            - "Affected features/components": The part of the application that is affected.
            - "Severity": The severity of the issue, categorized as 'Low', 'Medium', or 'High'.

            Your response MUST be ONLY the JSON object. Do NOT include any other text, explanations, or code block delimiters.
            """
            response_str = self.client.invoke_chat_model(prompt)
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from LLM (Attempt {attempt}): {e}")
            if attempt < 3:
                logging.info(f"Retrying to get a valid JSON response (Attempt {attempt + 1}).")
                return self._get_issue_summary_json(query, attempt + 1)
            else:
                return None

    def issue_summary_tool(self, query: str) -> Dict[str, Any]:
        """
        Tool to summarize a user-reported issue.
        It uses the LLM to extract key information like reported issues, affected components, and severity.
        """
        logging.info("Using issue_summary_tool")
        
        json_summary = self._get_issue_summary_json(query)
        
        if json_summary is not None:
            return json_summary
        else:
            return {"error": "Failed to generate a valid JSON summary after multiple attempts."}

    def internal_qa_tool(self, query: str) -> str:
        """
        Tool to answer questions based on internal documents.
        It searches the vector store for relevant information and uses the LLM to generate an answer.
        """
        if not self.vector_store:
            return "Vector store is not initialized. Please ensure documents were ingested correctly."

        try:
            logging.info("Using internal_qa_tool")
            retrieved_docs = self.vector_store.similarity_search(query, k=3)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""
            Answer the following question based only on the provided context.
            If the answer is not in the context, say "I can't find the answer to that question in the internal documents."

            Question: {query}
            
            Context:
            {context}

            Answer:
            """
            
            answer = self.client.invoke_chat_model(prompt)
            return answer

        except Exception as e:
            logging.error(f"Failed to use internal_qa_tool: {e}")
            return "An internal error occurred while retrieving or generating the answer."
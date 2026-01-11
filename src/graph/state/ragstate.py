#RAG state definition for LangGraph

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

from src.logger.logging import logging


class RAGState(BaseModel):
    logging.info("Initialize RAG State by Pydantic BaseModel")
    """State Object for RAG Workflow"""

    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
    
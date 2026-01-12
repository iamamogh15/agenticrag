#Configuration module for Agentic RAG System

import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from src.exception.exception import RAGException
from src.logger.logging import logging
from langchain_groq import ChatGroq
load_dotenv(override=True)

class Config:
    try:
        "Configure class for RAG System"
        #API keys
        logging.info("Set the key from .env")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        model = "llama3-70b-8192"

        # Default URLs
        Default_urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
        ]

        chunk_overlap: int = 50
        chunk_size: int = 500 

        @classmethod
        def get_llm(cls):
            """Initialize and return the LLM model"""
            return ChatGroq(
                api_key=cls.GROQ_API_KEY,model_provider="groq",
                model=cls.model)
    except Exception as e:
        raise RAGException(e,sys)

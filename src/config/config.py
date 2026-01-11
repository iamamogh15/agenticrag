#Configuration module for Agentic RAG System

import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from src.exception.exception import RAGException
from src.logger.logging import logging

load_dotenv(override=True)

class Config:
    try:
        "Configure class for RAG System"
        #API keys
        logging.info("Set the key from .env")
        groq_key = os.getenv("groq_key")
        model = "llama3-70b-8192"

        # Default URLs
        Default_urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
        ]

        @classmethod
        def get_llm(cls):
            """Initialize and return the LLM model"""
            os.environ["groq_key"] = cls.groq_key
            return init_chat_model(cls.model)
    except Exception as e:
        raise RAGException(e,sys)

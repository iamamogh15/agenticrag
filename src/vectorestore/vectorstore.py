#Vector Store module for document embedding and retrieval

import sys
from src.logger.logging import logging
from src.exception.exception import RAGException

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class Vectorestore:
    """Manages vectoreStore Operations"""
    def __init__(self):
        try:
            """initialize vector store with Hugging face"""
            logging.info("Initialize vector store")
            self.embedding = HuggingFaceEmbeddings()
            self.vectorstore = None
            self.retriever = None
        except Exception as e:
            raise RAGException(e,sys)
    
    def create_vectorstore(self, docs: List[Document]):
        try:
            """
            Create vectorstore from documents
            Args:
                Documents: List of documents to embedd
            """
            logging.info("def -> create_vectorstore, Creating vectore Store")
            self.vectorstore = FAISS.from_documents(docs, self.embedding)
            self.retriever = self.vectorstore.as_retriever()
        except Exception as e:
            raise RAGException(e,sys)
    
    def get_retriever(self):
        try:
            """
            Get the retriever instance
            Returns:
                Retriever Instance
            """
            logging.info("def -> get_retriever, Iitialize retriever blockage before retreiving ")
            if self.retriever is None:
                raise ValueError("Vector store not initialized. Call create_vectorstore first.")
            return self.retriever
        except Exception as e:
            raise RAGException(e,sys)   
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        try:
            """
            Retrieve relevant documents for a query
            Args:
                Query: Search query
                K: Number of documents to retrieve
            Returns:
                Outputs: List of relevant documents
            """
            logging.info("def -> retrieve,Invoking retrieve with top k search")
            if self.retriever is None:
                raise ValueError("Vector Store not initialized. Call create_vectorstore first.")
            return self.retriever.invoke(query)
        except Exception as e:
            raise RAGException(e,sys)
    
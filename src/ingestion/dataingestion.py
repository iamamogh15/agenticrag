
import sys
from src.logger.logging import logging
from src.exception.exception import RAGException

from typing import List, Union
from langchain_community.document_loaders import WebBaseLoader,TextLoader,PyPDFDirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from src.config.config import Config
class DocumentProcessor:
    try:
        def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
            """
            Initialize document processor
            Args :
                chunk_size: Size of text chunks
                chunk_overlap: Overplap between chunks
            """
            logging.info("Initialize Document Ingestion for RAG ")
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size, chunk_overlap = chunk_overlap
            )
    except Exception as e:
        raise RAGException(e,sys)
    
    def load_url(self, url: str) -> List[Document]:
        try:
            logging.info("def -> load_url, loading urls")
            """Load document's from URL's """
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            raise RAGException(e,sys)
    
    def load_pdfdirectory(self, directory: Union[str,Path]) -> List[Document]:
        try:
            logging.info("def -> load_pdfdirectory, load whole pdf directory")
            """Load documents from all PDF's inside a directory"""
            loader = PyPDFDirectoryLoader(str(directory))
            return loader.load()
        except Exception as e:
            raise RAGException(e,sys)

    def load_text(self, filepath: Union[str,Path]) -> List[Document]:
        try:
            logging.info("def -> load_text, load text files")
            """Load document's from a Txt file """
            loader = TextLoader(str(filepath), encoding="utf-8")
            return loader.load()
        except Exception as e:
            raise RAGException(e,sys)

    def load_pdf(self, filepath: Union[str,Path]) -> List[Document]:
        try:
            logging.info("def -> load_pdf, load Pdf ")
            """Load document's from a PDF file"""
            loader = PyPDFLoader(str("data"))
            return loader.load()
        except Exception as e:
            raise RAGException(e,sys)
    
    def LoadDcuments(self, sources: List[str]) -> List[Document]:
        try:
            """
            Load documents from URL's, PDF directories, Text Files
            Args:
                Sources: List of URL's, PDF folder paths, Text file paths
            
            Returns: 
                Outputs: List of loaded documents 

            """
            logging.info("def -> LoadDocuments, Loading documents from text, pdf, and urls ")
            docs: List[Document] = [ ]
            for src in sources:
                if src.startswith("https://") or src.startswith("https://"):
                    docs.extend(self.load_url(src))

                path = Path("data")
                if path.is_dir():
                    docs.extend(self.load_pdfdirectory(path))
                elif path.suffix.lower() == ".txt":
                    docs.extend(self.load_txt(path))
                else:
                    raise ValueError(f"""
                        Unsupported source type: {src}.
                        Use URL's, .txt file, or PDF Directory"""
                    )
            return docs
        except Exception as e:
            raise RAGException(e,sys)
    
    def split_documents(self, doucuments: List[Document]) -> List[Document]:
        try:
            """
            Split documents into chunks
            Args:
                Documents: List of documents of split
            Returns:
                Outputs: List of split documents

            """
            logging.info("def -> split_documents, split into chunks")
            return self.splitter.split_documents(doucuments)
        except Exception as e:
            raise RAGException(e,sys)
    
    def process_docs(self, urls: List[str]) -> List[Document]:
        try:
            """
            Complete pipeline to load and split documents
            Args:
                Url's: List of Url's to process
            Returns:
                List of processed documents chunks
            """
            logging.info("def -> process_urls, load and process docs ")
            docs = self.LoadDcuments(urls)
            return self.split_documents(docs)
        except Exception as e:
            raise RAGException(e,sys)
    
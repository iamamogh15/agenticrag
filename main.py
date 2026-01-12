
import sys
from pathlib import Path

from src.exception.exception import RAGException
from src.logger.logging import logging

from src.config.config import Config
from src.ingestion.dataingestion import DocumentProcessor
from src.vectorestore.vectorstore import Vectorestore
from src.graph.graphbuilder.graphbuilder import GraphBuilder

class AgenticRAG:
    "Agentic applicatio RAG"
    def __init__(self, urls = None):
        """
        Initilize Agentic RAG system
        Args:
            Url's: List of urls to process(uses defaults if there)
        
        """
        print("Initialize Agentic RAG System")

        #use default Url's if None provided
        self.urls = urls or Config.Default_urls

        #initialize components
        self.llm = Config.get_llm()
        self.doc_processor = DocumentProcessor(
            chunk_overlap= Config.chunk_overlap,
            chunk_size=Config.chunk_size
        )
        self.vectorstore = Vectorestore()

        #Process documents and create vector store
        self._setup_vectorstore()

        #build graph
        self.graph_builder = GraphBuilder(
            retriever= self.vectorstore.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()
        print("System initialized sucessfully\n")
    
    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""
        print(f"Processing {len(self.urls)} URLs...")
        documents = self.doc_processor.process_docs(self.urls)
        print(f"Created {len(documents)} document chunks")
        
        print("Creating vector store...")
        self.vectorstore.create_vectorstore(documents)
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system
        Args:
            Question: User question
        Returns:
            Output: Generated answer
        """
        print(f"Question: {question}\n")
        print("Processing...")

        result = self.graph_builder.run(question)
        answer = result['answer']
        
        print(f"LLM Answer: {answer}\n")
        return answer
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("Interactive Mode - Type 'quit' to exit\n")
        
        while True:
            question = input("Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question:
                self.ask(question)
                print("-" * 80 + "\n")

def main():
    # Example: Load URLs from file if exists
    urls_file = Path("data/urls.txt")
    urls = None
    
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    # Initialize RAG system
    rag = AgenticRAG(urls=urls)
    
    # Example questions
    example_questions = [
        "What is the concept of agent loop in autonomous agents?",
        "What are the key components of LLM-powered agents?",
        "Explain the concept of diffusion models for video generation."
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # Optional: Run interactive mode
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()

if __name__ == "__main__":
    main()

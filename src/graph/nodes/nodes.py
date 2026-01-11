#langGraph nodes for RAG workflow
import sys
from src.logger.logging import logging
from src.exception.exception import RAGException

from src.graph.state.ragstate import RAGState

class RAGNodes:
    try:
        logging.info("Initialize RAG nodes ")
        """Contains node functions for RAG workflow"""
        def __init__(self, retriever, llm):
            """
            Initialize RAG nodes
            Args:
                Retriever: Document retriever instance
                LLM: Language model instance
            """
            logging.info("Initialize RAG nodes")
            self.retriever = retriever
            self.llm = llm
    except Exception as e:
        raise RAGException(e,sys)
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        try:
            logging.info("def ->retrieve_docs, for normal RAG")
            """
            Retrieve relevant documents node
            Args:
                State: Current RAG state
            Returns:
                Updated RAG state with retrieved documents
            """
            logging.info("def -> retrieve_docs, retrieve docs from store by invoking question")
            docs = self.retriever.invoke(state.question)
            return RAGState(question=state.question,retrieved_docs=docs)
        except Exception as e:
            raise RAGException(e,sys)
    
    def generate_answer(self, state: RAGState) -> RAGState:
        try:
            """ 
            Generate answer from retrieved documents node
            Args:
                State: Current RAG state with retrieved documents
            Returns:
                Outputs: Updated RAG state with generated answer
            """
            logging.info("def -> generate_answers, generate answers/response from retrieved docs ")
            context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
            prompt = f"""Answer the question based on the context.
            Context: {context}
            Question: {state.question}
            """
            #Generate response
            response = self.llm.invoke(prompt)
            return RAGState(
                question = state.question,
                retrieved_docs=state.retrieved_docs,
                answer=response.content
            )
        except Exception as e:
            raise RAGException(e,sys)

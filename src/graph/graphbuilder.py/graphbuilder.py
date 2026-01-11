#Graph builder for LangGraph workflow

import sys
from src.logger.logging import logging
from src.exception.exception import RAGException

from langgraph.graph import StateGraph, END
from src.graph.state.ragstate import RAGState
from src.graph.nodes.reactnodes import RAGNodes

class GraphBuilder:
    try:
        """Builds and manages the LangGraph Workflow"""
        def __init__(self, retriever, llm):
            """
            Initialize graph builder
            Args:
            Retriever: Document retriever instance
            LLM: Language model instance
            """
            logging.info("Initialize Graph nodes and nodes")
            self.nodes = RAGNodes(retriever = retriever, llm = llm)
            self.graph = None
    except Exception as e:
        raise RAGException(e,sys)

    def build(self):
        try:

            """
            Build the RAG workflow graph
            Returns:
                Compiled graph instance
            """
            logging.info("def -> build, Adding graph nodes and edges")
            #create builder
            builder = StateGraph(RAGState)
            #Nodes
            builder.add_node("retriever",self.nodes.retrieve_docs)
            builder.add_node("responder", self.nodes.generate_answer)

            builder.set_entry_point("retriever")

            #edges
            builder.add_edge("retriever", "responder")
            builder.add_edge("responder", END)

            #Compile
            self.graph = builder.compile()
            return self.graph
        except Exception as e:
            raise RAGException(e,sys)
    
    def run(self, question: str) -> dict:
        try:

            """
            Run the RAG workflow
            Args:
                Question: User question
            Returns:
                Output: Final state with answer
            """
            logging.info("def -> run, Running Graph nodes and nodes")
            if self.graph is None:
                self.build()
            initial_state = RAGState(question = question)
            return self.graph.invoke(initial_state)
        except Exception as e:
            raise RAGException(e,sys)
        
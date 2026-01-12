#LangGraph nodes for RAG workflow + ReAct Agent generate_content

from typing import List, Optional
from src.graph.state.ragstate import RAGState

import sys
from src.logger.logging import logging
from src.exception.exception import RAGException

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

#Wikipedia Tools
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

class RAGNodes:
    try:
        """
        Contains node functions for RAG workflow
        """
        logging.info("Initialize RAG nodes for Re+Act")
        def __init__(self,retriever, llm):
            self.retriever = retriever
            self.llm = llm
            self._agent = None
    except Exception as e:
        raise RAGException(e,sys)

    def retrieve_docs(self, state: RAGState) ->RAGState:
        try:
            """Classic retriever node """
            logging.info("def -> retrieve docs for Re+Act ")
            docs = self.retriever.invoke(state.question)
            return RAGState(
                question=state.question,
                retrieved_docs=docs
            )
        except Exception as e:
            raise RAGException(e,sys)
        
    def _built_tools(self) -> list[Tool]:
        try:
            logging.info("Building Tools for Re+Act")
            """Build retriever + wikipedia Tools"""

            def retriever_tool(query: str) -> str:
                try:
                    logging.info("def -> retrieve_tool, retrieve tools for docs,and know there metadata")
                    docs: List[Document] = self.retriever.invoke(query)
                    if not docs:
                        return "No document found. "
                    merged = []
                    for key, value in enumerate(docs[:8],start = 1):
                        meta = value.metadata if hasattr(value, "metadata") else {}
                        title = meta.get("title") or meta.get("source") or f"doc_{key}"
                        merged.append(f"[{key}]{title}\n{value.page_content}")
                    return "\n\n".join(merged)
                except Exception as e:
                    raise RAGException(e,sys)
            logging.info("Retrieve tool -> retrieving docs from vector store")
            retriever_tool = Tool(
                name = "retriever",
                description = "fetch passages from indexed corpus",
                func = retriever_tool,
            )
            logging.info("Retrieve tool -> wikipedia")
            wiki = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(top_k_results = 3, lang = "en")
            )
            wikipedia_tool = Tool(
                name = "wikipedia",
                description= "Search wikipedia for general knowledge.",
                func= wiki.run,
            )

            return [retriever_tool, wikipedia_tool]
        except Exception as e:
            raise RAGException(e,sys)
    
    def _build_agent(self):
        try:
            logging.info("def -> Bilding Re+Act agent")
            """React Agent with Tools"""
            tools = self._build_agent()
            system_prompt = (
                "you are a helpful RAG agent. "
                "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
                "Return only the final useful answer."
            )
            self._agent = create_react_agent(
                self.llm, tools=tools,
                prompt = system_prompt
            )
        except Exception as e:
            raise RAGException(e,sys)

    def generate_answer(self, state: RAGState) -> RAGState:
        try:
            logging.info("def -> generate answer, for Re+Act")
            """Generate answer using ReAct agent with retriever + wikipedia tool """
            if self._agent is None:
                self._build_agent()
            
            result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
            
            messages = result.get("messages", [])
            answer: Optional[str] = None
            if messages:
                answer_message = messages[-1]
                answer = getattr(answer_message, "content", None)
            
            return RAGState(
                question = state.question,
                retrieved_docs = state.retrieved_docs,
                answer = answer or "Could not generate answer."
            )
        except Exception as e:
            raise RAGException(e,sys)

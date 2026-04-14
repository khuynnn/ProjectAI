import asyncio
import os
from typing import Any, Annotated, TypedDict, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from src.rag.retrieval.retriever import Retriever
from src.utils.log import get_logger

from src.rag.generation.prompts import (RAG_PROMPT_TEMPLATE, SUMMARIZE_PROMPT_TEMPLATE, SUMMARIZE_INIT_PROMPT)

load_dotenv()
logger = get_logger(__name__)


class LMState(TypedDict):
    """State for LM conversation with message memory."""
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str] = None
    context: Optional[str] = None


class LMClient:
    """Wrapper for calling the DeepSeek chat LLM with memory via langgraph."""

    PROMPT_TEMPLATE = RAG_PROMPT_TEMPLATE

    def __init__(
        self, 
        model_name: str = os.getenv("LLM_NAME"), 
        temperature: float = 0.0
        ):

        self.model_name = model_name
        self.temperature = temperature
        logger.info("Initializing LMClient(model_name=%s, temperature=%s)", self.model_name, self.temperature)

        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE, verbose=True)
        self.llm = ChatDeepSeek(model=self.model_name, temperature=self.temperature)
        self.chain = self.prompt | self.llm
        
        # Build langgraph with memory
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)

        self.retriever = Retriever()

        logger.info("LMClient initialized successfully with langgraph memory and checkpointer")

    async def build_context(self, query: str) -> str:
        """Build context text with metadata from retriever results."""
        reranked = await self.retriever.query_and_rerank(query)
        
        context_parts = []
        for i, doc in enumerate(reranked, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            source   = metadata.get("source", metadata.get("file_name", f"Tài liệu {i}"))
            chapter  = metadata.get("chapter", "")
            page     = metadata.get("page", "")

            header_parts = [f"[Nguồn: {source}"]
            if chapter:
                header_parts.append(f"Chương: {chapter}")
            if page:
                header_parts.append(f"Trang: {page}")
            header = " | ".join(header_parts) + "]"

            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            context_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(context_parts)

    async def retrieve_context_node(self, state: LMState) -> dict:
        """Retrieve or reuse context and attach it to the message history."""
        messages = state["messages"]
        user_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        question = user_msg.content if user_msg else ""
        context_text = state.get("context", "")

        if not context_text and question:
            context_text = await self.build_context(question)

        system_msg = SystemMessage(content=self.PROMPT_TEMPLATE.format(context=context_text, question=""))
        new_messages = [system_msg]
        if user_msg:
            new_messages.append(user_msg)

        return {"messages": new_messages}

    # Add node for LLM call
    async def call_lm(self, state: LMState) -> dict:
        """Call LLM with entire message history."""
        messages = state["messages"]
        
        if not messages:
            logger.warning("No messages in state, returning empty response")
            return {"messages": []}
        
        logger.info("Invoking LLM with %d messages in history", len(messages))
        result = await self.llm.ainvoke(messages)
        logger.info("LLM response received")
        
        return {"messages": [result]}
    
    # Add summarize node
    async def summarize_conversation(self, state: LMState) -> dict:
        """Summarize the conversation and delete old messages."""
        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt
        if summary:
            # A summary already exists
            summary_message = SUMMARIZE_PROMPT_TEMPLATE.format(summary=summary)
        else:
            summary_message = SUMMARIZE_INIT_PROMPT

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = await self.llm.ainvoke(messages)

        # Keep only the 2 most recent messages to keep history compact
        trimmed_messages = state["messages"][-2:]
        return {"summary": response.content, "messages": trimmed_messages}
    
    # Conditional: if messages > 6, summarize, else end
    def should_summarize(self, state: LMState) -> str:
        messages = state["messages"]
        if len(messages) > 6:
            return "summarize_node"
        return END

    def _build_graph(self):
        """Build a langgraph StateGraph with message memory."""
        graph = StateGraph(LMState)

        graph.add_node("context_node", self.retrieve_context_node)
        graph.add_node("lm_node", self.call_lm)
        graph.add_node("summarize_node", self.summarize_conversation)
        
        graph.add_edge(START, "context_node")
        graph.add_edge("context_node", "lm_node")
        graph.add_conditional_edges("lm_node", self.should_summarize)
        graph.add_edge("summarize_node", END)
        
        return graph

    async def ask(self, question: str, context: Any = None, thread_id: str = "default") -> str:
        if isinstance(context, str):
            context_text = context
        elif context is None:
            context_text = ""
        else:
            context_text = self.retriever.combine_docs(context)

        logger.info("Asking LM question=%s in thread=%s", question, thread_id)
        logger.debug("Context length=%d characters", len(context_text))

        user_msg = HumanMessage(content=question)
        initial_state = {"messages": [user_msg], "context": context_text}
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.compiled_graph.ainvoke(initial_state, config=config)
            
        # Extract answer from last message
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            answer = getattr(last_msg, "content", str(last_msg))
            logger.info("LM returned answer with %d characters", len(answer))
            return answer
        
        logger.warning("No response from LM graph")
        return ""

async def main():
    query = "Công thức hàm đối ngẫu Lagrange là gì"
    lm_client = LMClient()

    answer_text = await lm_client.ask(query, thread_id="main_test")
    print(answer_text)

if __name__ == "__main__":
    asyncio.run(main())

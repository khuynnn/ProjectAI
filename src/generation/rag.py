import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import Annotated, TypedDict

from src.retrieval.retriever import Retriever
from src.utils.log import get_logger

load_dotenv()
logger = get_logger(__name__)


class LMState(TypedDict):
    """State for LM conversation with message memory."""
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    context: str


class LMClient:
    """Wrapper for calling the DeepSeek chat LLM with memory via langgraph."""

    PROMPT_TEMPLATE = """
    You are an AI assistant that answers questions based ONLY on the provided context.

    GENERAL RULES:
    - Use only the information from the CONTEXT.
    - If the answer is not in the context, say: "I don't have enough information."
    - Be clear, accurate, and well-structured.
    - Avoid unnecessary repetition or verbosity.

    ADAPTIVE RESPONSE STYLE:
    - If the question asks for a definition → give a concise definition first, then explain.
    - If the question involves formulas or mathematics → include properly formatted LaTeX using $$ $$.
    - If the question is conceptual → provide a clear explanation with logical structure.
    - If the question is procedural → answer step-by-step.
    - If multiple points are needed → use bullet points.

    FORMATTING:
    - Use paragraphs for explanations.
    - Use bullet points when listing ideas.
    - Use LaTeX only when necessary (not for every answer).

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

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

    def build_context(self, query: str) -> str:
        """Build combined context text from retriever results."""
        reranked = self.retriever.query_and_rerank(query, k=5)
        return self.retriever.combine_docs(reranked)

    def retrieve_context_node(self, state: LMState) -> dict:
        """Retrieve or reuse context and attach it to the message history."""
        messages = state["messages"]
        user_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
        question = user_msg.content if user_msg else ""
        context_text = state.get("context", "")

        if not context_text and question:
            context_text = self.build_context(question)

        system_msg = SystemMessage(content=self.PROMPT_TEMPLATE.format(context=context_text, question=""))
        new_messages = [system_msg]
        if user_msg:
            new_messages.append(user_msg)

        return {"messages": new_messages}

    def _build_graph(self):
        """Build a langgraph StateGraph with message memory."""
        graph = StateGraph(LMState)
        
        # Add node for LLM call
        def call_lm(state: LMState) -> dict:
            """Call LLM with entire message history."""
            messages = state["messages"]
            
            if not messages:
                logger.warning("No messages in state, returning empty response")
                return {"messages": []}
            
            logger.info("Invoking LLM with %d messages in history", len(messages))
            result = self.llm.invoke(messages)
            logger.info("LLM response received")
            
            return {"messages": [result]}
        
        # Add summarize node
        def summarize_conversation(state: LMState) -> dict:
            """Summarize the conversation and delete old messages."""
            # First, we get any existing summary
            summary = state.get("summary", "")

            # Create our summarization prompt
            if summary:
                # A summary already exists
                summary_message = (
                    f"This is a summary of the conversation to date: {summary}\n\n"
                    "Extend the summary by taking into account the new messages above:"
                )
            else:
                summary_message = "Create a summary of the conversation above:"

            # Add prompt to our history
            messages = state["messages"] + [HumanMessage(content=summary_message)]
            response = self.llm.invoke(messages)

            # Keep only the 2 most recent messages to keep history compact
            trimmed_messages = state["messages"][-2:]
            return {"summary": response.content, "messages": trimmed_messages}
        
        # Conditional: if messages > 6, summarize, else end
        def should_summarize(state: LMState) -> str:
            messages = state["messages"]
            if len(messages) > 6:
                return "summarize_node"
            return END

        graph.add_node("context_node", self.retrieve_context_node)
        graph.add_node("lm_node", call_lm)
        graph.add_node("summarize_node", summarize_conversation)
        
        graph.add_edge(START, "context_node")
        graph.add_edge("context_node", "lm_node")
        graph.add_conditional_edges("lm_node", should_summarize)
        graph.add_edge("summarize_node", END)
        
        return graph

    def ask(self, question: str, context: Any = None, thread_id: str = "default") -> str:
        if isinstance(context, str):
            context_text = context
        elif context is None:
            context_text = ""
        else:
            context_text = self.retriever.combine_docs(context)

        logger.info("Asking LM question=%s in thread=%s", question, thread_id)
        logger.debug("Context length=%d characters", len(context_text))

        user_msg = HumanMessage(content=question)
        initial_state = {"messages": [user_msg], "summary": "", "context": context_text}
        config = {"configurable": {"thread_id": thread_id}}
        result = self.compiled_graph.invoke(initial_state, config=config)
        
        # Extract answer from last message
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            answer = getattr(last_msg, "content", str(last_msg))
            logger.info("LM returned answer with %d characters", len(answer))
            return answer
        
        logger.warning("No response from LM graph")
        return ""


if __name__ == "__main__":
    query = "Công thức hàm đối ngẫu Lagrange là gì"
    lm_client = LMClient()

    answer_text = lm_client.ask(query, thread_id="main_test")
    print(answer_text)

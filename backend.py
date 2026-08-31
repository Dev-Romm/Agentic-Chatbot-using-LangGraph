from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

import requests
import math
import os

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-5-mini",
    openai_api_version="2025-04-01-preview"
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_deployment="text-embedding-3-small",
    api_version="2023-05-15"
)

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
   
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )

    return retriever

# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using thr stored PDF documents.

    Args:
        query: the question or search used to retrieve the PDF content.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page {page}\n"
            f"Content {document.page_content}\n"
        )

    return "\n\n".join(formatted_documents)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return str(result)
    except (SyntaxError, TypeError, NameError, ValueError, ZeroDivisionError) as error:
        return f"Unable to calculate expression: {error}"

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                    "Nairobi", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return "Weather service is not configured."

    try:
        response = requests.get(
            "https://api.weatherstack.com/current",
            params={"access_key": api_key, "query": location},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            return f"Unable to get weather: {data['error'].get('info', 'unknown error')}"

        current = data.get("current", {})
        descriptions = current.get("weather_descriptions") or ["unknown conditions"]
        place = data.get("location", {}).get("name", location)
        return (
            f"Weather in {place}: {current.get('temperature', 'unknown')}°C, "
            f"{descriptions[0]}, humidity {current.get('humidity', 'unknown')}%."
        )
    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        return f"Unable to get weather: {error}"
    
# Bind tools to LLM
tools = [search_tool, get_current_weather, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)
llm_with_required_tools = llm.bind_tools(tools, tool_choice="required")


def requires_live_search(messages: list[BaseMessage]) -> bool:
    """Identify requests where an answer should be based on current information."""
    latest_message = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )
    if latest_message is None:
        return False

    query = str(latest_message.content).lower()
    search_terms = (
        "best movies", "latest", "current", "recent", "upcoming", "today", "this year"
    )
    return any(term in query for term in search_terms)

# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node 1
def chat_node(state: ChatState):
    """LLM node that can answer directlybor call an appropriate tool."""
    system_message = SystemMessage(
        content=(
            "You are a helpful, accurate assistant with access to four tools: "
            "search, get_current_weather, calculator, and rag_tool. Choose the "
            "most appropriate tool whenever it can improve accuracy, and do not "
            "claim to have used a tool when you did not.\n\n"
            "Tool guidance:\n"
            "- Use search for current, recent, upcoming, time-sensitive, or "
            "real-world information, news, recommendations, comparisons, prices, "
            "events, and questions such as the best movies of a particular year. "
            "Use the search results as evidence, distinguish facts from opinions, "
            "and avoid presenting unverified claims as certain.\n"
            "- Use get_current_weather for present weather conditions at a specified "
            "city or location. Do not use search as a substitute when the user asks "
            "for current weather. If the location is ambiguous, ask for clarification.\n"
            "- Use calculator for arithmetic, numerical expressions, and simple math "
            "functions. Pass only a valid expression, then report the result clearly "
            "and preserve meaningful units or assumptions.\n"
            "- Use rag_tool for factual or conceptual questions that may be answered "
            "from the stored PDF documents. Prefer it when the user refers to the "
            "document, its contents, or information expected to come from the PDF. "
            "Base the answer only on the retrieved passages, acknowledge when the "
            "documents do not contain enough information, and include relevant source "
            "or page details when available.\n\n"
            "Use more than one tool when a request genuinely requires multiple kinds "
            "of information, but avoid unnecessary tool calls. After a tool returns, "
            "interpret its output and answer the user's question directly. Never "
            "invent search results, PDF content, calculations, weather data, or "
            "sources. The current date is August 2026; do not claim that 2026 is in "
            "the future."
        )
    )
    messages = state['messages']
    # send to llm
    messages_with_instructions = [system_message, *messages]
    model = llm_with_required_tools if requires_live_search(messages) else llm_with_tools
    response = model.invoke(messages_with_instructions)
    # response store state
    return {'messages': [response]}

# Node 2
tool_node = ToolNode(tools)


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])

    return list(all_threads)
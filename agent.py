import os
import csv
import math
import pandas as pd
import numpy as np

# --- LangChain imports (latest structure) ---
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationSummaryMemory
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

# --- Environment API keys ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Set the environment variable OPENAI_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Set the environment variable TAVILY_API_KEY")

# --- OpenAI client ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# --- Embeddings and FAISS vector store ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# --- Tools ---
amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search for information about Amazon products.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    return amazon_tool.run(query)

# Tavily search tool
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search_tool = TavilySearchResults(max_results=5, include_images=True)

@tool
def search_tavily(query: str):
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# --- Prompt template ---
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the user's queries based on available tools.

User Input: {input}"""
)

# --- Conversation memory ---
summary_memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500,
    memory_key="chat_history",
    return_messages=True
)

# --- Agent ---
summary_react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

summary_agent_executor = AgentExecutor(
    agent=summary_react_agent,
    tools=tools,
    memory=summary_memory,
    verbose=True
)

# --- Gradio app ---
import gradio as gr

# Per-session memory
session_memory = {}

def get_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
    return session_memory[session_id]

def chat_with_agent(user_input, session_id):
    memory = get_memory(session_id)
    agent_executor = AgentExecutor(
        agent=summary_react_agent,
        tools=tools,
        memory=memory,
        verbose=True
    )
    response = agent_executor.invoke({"input": user_input})
    retur

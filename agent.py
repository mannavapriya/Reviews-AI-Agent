import os
import csv
import math
import numpy as np
import pandas as pd

# -----------------------------
# LangChain Core imports
# -----------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryMemory
from langchain import hub

import gradio as gr

# -----------------------------
# API Keys (set directly for EC2)
# -----------------------------
OPENAI_API_KEY = "your_openai_api_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# -----------------------------
# Embeddings + FAISS Vector Store
# -----------------------------
embeddings = OpenAIEmbeddings()

vector = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -----------------------------
# Amazon Retriever Tool
# -----------------------------
amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search FAISS index for Amazon product data.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    """
    Search for information about Amazon products using FAISS index.
    """
    return amazon_tool.run(query)

# -----------------------------
# Tavily Search Tool
# -----------------------------
tavily_search_tool = TavilySearchResults(
    max_results=5,
    include_images=True
)

@tool
def search_tavily(query: str):
    """
    Executes a web search using TavilySearchResults tool.
    Returns a list of search results with answers and images.
    """
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# -----------------------------
# Prompt from LangChain Hub
# -----------------------------
prompt = hub.pull("hwchase17/react")

# -----------------------------
# Memory
# -----------------------------
summary_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True
)

session_memory = {}

def get_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=summary_llm,
            memory_key="chat_history",
            return_messages=True
        )
    return session_memory[session_id]

# -----------------------------
# ReAct Agent
# -----------------------------
summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=tools,
    prompt=prompt
)

# -----------------------------
# Agent execution function
# -----------------------------
def chat_with_agent(user_input, session_id):
    memory = get_memory(session_id)

    agent_executor = AgentExecutor(
        agent=summary_react_agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    response = agent_executor.invoke({"input": user_input})

    if isinstance(response, dict) and "output" in response:
        return response["output"]
    return response

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie — LangChain ReAct Agent")
    gr.Markdown("Ask anything. Session-based memory enabled.")

    with gr.Row():
        input_box = gr.Textbox(label="Your question:")
        output_box = gr.Textbox(label="AI Response:", lines=10)

    submit = gr.Button("Submit")
    submit.click(
        chat_with_agent,
        inputs=[input_box, gr.State("session")],
        outputs=output_box
    )

app.launch(debug=True, share=True)

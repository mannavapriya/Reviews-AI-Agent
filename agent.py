import os
import csv
import math
import numpy as np
import pandas as pd

# -----------------------------
# LangChain imports
# -----------------------------
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool, VectorStoreRetrieverTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationSummaryMemory
from langchain import hub  # For pulling prompts from LangChain Hub

# -----------------------------
# Gradio
# -----------------------------
import gradio as gr

# -----------------------------
# API Keys (set these on your EC2 environment)
# Example: export OPENAI_API_KEY="sk-xxxx" 
#          export TAVILY_API_KEY="tvly-xxxx"
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY and TAVILY_API_KEY in your environment")

# -----------------------------
# Embeddings + FAISS vector store
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
amazon_tool = VectorStoreRetrieverTool(
    name="Amazon Product Search",
    description="Search FAISS index for Amazon product data.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
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
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# -----------------------------
# Load ReAct prompt from LangChain Hub
# -----------------------------
prompt = hub.pull("hwchase17/react")

# -----------------------------
# Chat LLM
# -----------------------------
summary_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True
)

# -----------------------------
# Session memory
# -----------------------------
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
        input_box = gr.Textbox(label="Your question:", placeholder="Type your query here...")
        output_box = gr.Textbox(label="AI Response:", lines=10)

    submit = gr.Button("Submit")
    submit.click(
        chat_with_agent,
        inputs=[input_box, gr.State("session")],
        outputs=output_box
    )

app.launch(debug=True, share=True)

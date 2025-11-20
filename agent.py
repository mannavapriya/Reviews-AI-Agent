import os
import csv
import math
import numpy as np
import pandas as pd

# -----------------------------
# LangChain imports (modern structure)
# -----------------------------
from langchain.prompts.chat import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.tools import tool, VectorStoreRetrieverTool
from langchain.memory import ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentExecutor, AgentType
from langchain import hub  # For pulling prompts from LangChain Hub

# -----------------------------
# Gradio
# -----------------------------
import gradio as gr

# -----------------------------
# API Keys (set these in EC2 environment)
# -----------------------------
# Make sure you export these in your EC2 shell:
# export OPENAI_API_KEY="your_openai_key"
# export TAVILY_API_KEY="your_tavily_key"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# -----------------------------
# Embeddings + FAISS vector store
# -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
    """
    Search for information about Amazon products.
    """
    return amazon_tool.run(query)

# -----------------------------
# Tavily Search Tool
# -----------------------------
# You may need to reinstall langchain-community
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search_tool = TavilySearchResults(
    max_results=5,
    include_images=True,
    tavily_api_key=TAVILY_API_KEY
)

@tool
def search_tavily(query: str):
    """
    Search the web using Tavily.
    """
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
    model_name="gpt-4o-mini",
    temperature=0,
    streaming=True,
    openai_api_key=OPENAI_API_KEY
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
summary_react_agent = initialize_agent(
    tools=tools,
    llm=summary_llm,
    agent=AgentType.CHAT_REACT_DESCRIPTION,
    verbose=True,
    memory=None
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

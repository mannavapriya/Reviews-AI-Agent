import os
import csv
import math
import numpy as np
import pandas as pd

# LangChain modern imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI LLM + embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

# Agent & memory (modern)
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationSummaryMemory

# LangChain Hub (modern path)
from langchain_hub import pull_prompt

# -----------------------------
# API Keys
# -----------------------------
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# -----------------------------
# Embeddings + FAISS
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
# Prompt (via LC Hub)
# -----------------------------
prompt = pull_prompt("hwchase17/react")

# -----------------------------
# Session Memory
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
# Agent Execution Function
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
import gradio as gr

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

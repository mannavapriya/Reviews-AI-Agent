import os
import csv
import math
import pandas as pd
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryMemory
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub

import gradio as gr

# ----------------------
# API KEYS
# ----------------------
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# ----------------------
# Embeddings & VectorStore
# ----------------------
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ----------------------
# Tools
# ----------------------
# Amazon product search tool
amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search for information about Amazon products. Use for any questions about Amazon products.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    return amazon_tool.run(query)

# Tavily search tool
tavily_search_tool = TavilySearchResults(max_results=5, include_images=True)

@tool
def search_tavily(query: str):
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# ----------------------
# Load LangChain Hub Prompt
# ----------------------
prompt = hub.pull("hwchase17/react")

# ----------------------
# Session Memory
# ----------------------
summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

session_memory = {}

def get_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=summary_llm,
            memory_key="chat_history",
            return_messages=True
        )
    return session_memory[session_id]

# ----------------------
# Agent Setup
# ----------------------
summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=tools,
    prompt=prompt
)

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

# ----------------------
# Gradio UI
# ----------------------
with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_agent, inputs=[input_box, gr.State(value="session")], outputs=output_box)

# Launch app
app.launch(debug=True, share=True)

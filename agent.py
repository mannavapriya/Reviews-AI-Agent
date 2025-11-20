import os
import pandas as pd
import numpy as np
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationSummaryMemory
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

# ---------------------------
# Setup API Keys
# ---------------------------
Open_API_Key = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# LLM models
MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model_name=MODEL)

# ---------------------------
# Embeddings and Vectorstore
# ---------------------------
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ---------------------------
# Tools
# ---------------------------
amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search for information about Amazon products.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    return amazon_tool.run(query)

# Tavily search
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search_tool = TavilySearchResults(max_results=5, include_images=True)

@tool
def search_tavily(query: str):
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# ---------------------------
# Agent with Memory
# ---------------------------
prompt = hub.pull("hwchase17/react")

summary_memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=500,
    memory_key="chat_history",
    return_messages=True
)

react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=summary_memory,
    verbose=True
)

# ---------------------------
# Gradio App
# ---------------------------
import gradio as gr

session_memory = {}

def get_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
    return session_memory[session_id]

def chat_with_agent(user_input, session_id="default"):
    memory = get_memory(session_id)
    executor = AgentExecutor(agent=react_agent, tools=tools, memory=memory, verbose=True)
    response = executor.invoke({"input": user_input})
    return response.get("output", str(response))

with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_agent, inputs=input_box, outputs=output_box)

app.launch(debug=True, share=True)

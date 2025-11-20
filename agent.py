import os
import csv
import math
import pandas as pd
import numpy as np

# ==== LATEST LANGCHAIN IMPORTS ====
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_community.vectorstores import FAISS
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ==== ENVIRONMENT VARIABLES ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY env variable")
if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY env variable")

# ==== LLM ====
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

# ==== VECTOR STORE ====
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vector = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector.as_retriever(
    search_kwargs={"k": 4}
)

# ==== RETRIEVER TOOL ====
@tool
def amazon_product_search(query: str):
    """Search Amazon product info from FAISS index."""
    docs = retriever.get_relevant_documents(query)
    return "\n".join([d.page_content for d in docs])

# ==== TAVILY TOOL ====
tavily_search_tool = TavilySearchResults(
    max_results=5,
    include_images=True,
    tavily_api_key=TAVILY_API_KEY
)

@tool
def search_tavily(query: str):
    """Search the web using Tavily."""
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# ==== PROMPT ====
prompt = ChatPromptTemplate.from_template("""
You are ReviewGenie, an intelligent product-analysis assistant.
Use tools when beneficial.

User Input: {input}
""")

# ==== MEMORY ====
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# ==== AGENT ====
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    memory=summary_memory
)

# ==== GRADIO UI ====
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

def chat_with_agent(user_input, session_id):
    memory = get_memory(session_id)

    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        memory=memory
    )

    result = executor.invoke({"input": user_input})
    return result.get("output", result)

with gr.Blocks() as app:
    gr.Markdown("# 🤖 ReviewGenie — AI Product Research Agent")

    with gr.Row():
        inp = gr.Textbox(label="Ask a question")
        out = gr.Textbox(label="Response", lines=8)

    hidden_session = gr.Textbox(value="default_session", visible=False)

    gr.Button("Submit").click(
        chat_with_agent, 
        inputs=[inp, hidden_session],
        outputs=out
    )

app.launch(debug=True, share=True)

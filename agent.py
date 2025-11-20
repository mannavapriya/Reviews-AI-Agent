import os
import gradio as gr
import pandas as pd
import numpy as np
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryMemory
from langchain_community.tools.tavily_search import TavilySearchResults

# ======== SET UP API KEYS ========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======== LOAD EMBEDDINGS & FAISS VECTORSTORE ========
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ======== CREATE TOOLS ========
amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search for information about Amazon products. For any questions related to Amazon products, use this tool.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    return amazon_tool.run(query)

tavily_search_tool = TavilySearchResults(
    max_results=5,
    include_images=True
)

@tool
def search_tavily(query: str):
    return tavily_search_tool.run(query)

tools = [amazon_product_search, search_tavily]

# ======== DEFINE PROMPT ========
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use the following input to answer:
{input}"""
)

# ======== SET UP MEMORY ========
summary_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, streaming=True)

summary_memory = ConversationSummaryMemory(
    llm=summary_llm,
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=500
)

# ======== CREATE REACT AGENT ========
summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=tools,
    prompt=prompt
)

summary_agent_executor = AgentExecutor(
    agent=summary_react_agent,
    tools=tools,
    memory=summary_memory,
    verbose=True
)

# ======== GRADIO INTERFACE ========
session_memory = {}

def get_memory(session_id):
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=summary_llm,
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
    if isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        return response

with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_agent, inputs=[input_box, gr.State(value="session1")], outputs=output_box)

# Launch the Gradio app
app.launch(debug=True, share=True)

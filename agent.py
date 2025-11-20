import csv
import pandas as pd
import math
import numpy as np
import os
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_functions_agent
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

from langchain import hub # Used to pull predefined prompts from LangChain Hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory # Used to store chat history in memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

embeddings = OpenAIEmbeddings()
vector = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)

retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

amazon_tool = create_retriever_tool(
    name="Amazon Product Search",
    description="Search for information about Amazon products. For any questions related to Amazon products, use this tool.",
    retriever=retriever
)

@tool
def amazon_product_search(query: str):
    """
    Search for information about Amazon products.
    For any questions related to Amazon products, this tool must be used.
    """
    return amazon_tool.run(query)

from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search_tool = TavilySearchResults(
    max_results=5,        # maximum number of results to return
    include_images=True   # include image URLs in the results
)

@tool
def search_tavily(query: str):
    """
    Executes a web search using the TavilySearchResults tool.

    Parameters:
        query (str): The search query entered by the user.

    Returns:
        list: A list of search results containing answers, raw content, and images.
    """
    results = tavily_search_tool.run(query)
    return results

from langchain import hub

prompt = hub.pull("hwchase17/react")

tools = [amazon_product_search, search_tavily]

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=500,
    memory_key="chat_history",
    return_messages=True
)

summary_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, streaming=True)

from langchain.agents import create_react_agent

summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=[amazon_product_search, search_tavily],
    prompt=prompt
)

summary_agent_executor = AgentExecutor(
    agent=summary_react_agent,
    tools=[amazon_product_search, search_tavily],
    memory=summary_memory,
    verbose=True
)

import gradio as gr

session_memory = {}

def get_memory(session_id):
    """
    Fetch or create a ConversationSummaryMemory instance for a given session.
    """
    if session_id not in session_memory:
        session_memory[session_id] = ConversationSummaryMemory(
            llm=summary_llm,
            memory_key="chat_history",
            return_messages=True
        )
    return session_memory[session_id]

def chat_with_agent(user_input, session_id):
    """Processes user input and maintains session-based chat history."""

    memory = get_memory(session_id)

    # Create a per-session AgentExecutor
    agent_executor = AgentExecutor(
        agent=summary_react_agent,
        tools=[amazon_product_search, search_tavily],
        memory=memory,
        verbose=True
    )

    # Use invoke instead of run
    response = agent_executor.invoke({"input": user_input})

    # The output is typically under 'output' key
    if isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        # Sometimes the response might just be a string
        return response

with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")

    submit_button.click(chat_with_agent, inputs=input_box, outputs=output_box)

# Launch the Gradio app
app.launch(debug=True, share=True)
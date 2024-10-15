import random
import gradio as gr

from dotenv import load_dotenv
import openai
import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    PromptTemplate
)
load_dotenv("4.nlp/4.4.rag/.env")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
Settings.embed_model = embed_model

PERSIST_DIR = "4.nlp/4.4.rag/lyrics_store2"
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR) 
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(verbose=True)

qa_template_str = """
    You are an expert in Bolivian Folk music, your task is to guide and teach the user 
    about your field. Answer the user queries only with supported data in your context.
    Your context may contain complete lyrics or parts of them in different languages, but
    your answer will always be in Spanish. 

    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, 
    answer the query with detailed source information, include direct quotes and use bullet lists in your 
    answers, in one of the bullets detail the tone/sentiment of the song.
    Query: {query_str}
    Answer: 
"""
qa_template = PromptTemplate(qa_template_str)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_template}
)

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
description = """
A set of lyrics for songs from the Bolivian Folk Group Los Kjarkas. 
Use plain text question as input to the tool. 
MANDATORY: Pass the response to the user as is, mantaining the format, do not try to summarize when using this tool.
"""
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="Kjarkas_songs_lyrics",
            description=description,
            return_direct=False
        )
    )
]
agent = OpenAIAgent.from_tools(tools=tools, verbose=True)

def very_wise_response(message, history):
    return agent.chat(message).response

demo = gr.ChatInterface(very_wise_response, type="messages")

demo.launch()
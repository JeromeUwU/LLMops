import streamlit as st
import os
from langchain_community.llms import Ollama 
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS,Chroma
from dotenv import load_dotenv

load_dotenv()
groq_api_key="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"



if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = 'llama3.1')
    st.session_state.loader = PyPDFLoader("mg_pdf.pdf")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("Traduction Francais-Malgache")
llm = ChatOllama(model="llama3.1")

prompt = ChatPromptTemplate.from_template(
""" 
Find The Best Traduction for a word in french to malagasy from the context only.
The user will provid you with a french word and you have to translate it to malagasy only using the context.
<context>
{context}
Word : {input} 

"""
)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Your French word to translate to malagasy : ")

if prompt:
    response = retriever_chain.invoke({"input":prompt})
    st.write(response['answer'])
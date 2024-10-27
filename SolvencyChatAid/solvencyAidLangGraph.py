import streamlit as st
import os
from langchain_community.llms import Ollama 
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS,Chroma,Cassandra
from pydantic import BaseModel, Field
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langgraph.graph import END, StateGraph, START

from langchain.schema import Document
from typing import Literal,List
from typing_extensions import TypedDict


class RouteQuery(BaseModel):

    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

class GraphState(TypedDict):

    question: str
    generation: str
    documents: List[str]

def retrieve(state):
    question = state["question"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def wiki_search(state):
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki.invoke({"query": question})

    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)

    return {"documents": wiki_results, "question": question}

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        return "wiki_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"


if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model = 'llama3.1')
    st.session_state.loader = PyPDFDirectoryLoader("./solvencypdf/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 500, chunk_overlap = 0)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
    st.session_state.vectors=Chroma.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("SolvencyAid")
llm = ChatOllama(model="llama3.1")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to Solvency 2.
Use the vectorstore for questions on these topics and if you don't find satisfying response use wiki search.
Otherwise for any other question use wiki-search."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

retriever = st.session_state.vectors.as_retriever()

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)


workflow = StateGraph(GraphState)
workflow.add_node("wiki_search", wiki_search)  
workflow.add_node("retrieve", retrieve) 
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge( "retrieve", END)
workflow.add_edge( "wiki_search", END)

graph = workflow.compile()

prompt = st.text_input("Your question about Solvency 2 : ")

if prompt:
    response = graph.stream({
    "question": prompt
    })

    for output in response:
        for key, value in output.items():
            if key == 'retrieve':
                st.write(value['documents'][0].dict()['page_content'])
            if key == 'wiki_search':
                st.write(value['documents'].dict()['page_content'])
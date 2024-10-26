from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn 
import os 
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] =  os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title = "Server",
    version = "1.0",
    description = 'API',
)

llm = Ollama(model = 'llama3.1')

prompt = ChatPromptTemplate.from_template("Write a basic summary on this character {topic} with 100 words max")

add_routes(
    app,
    prompt|llm,
    path = "/summary",
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
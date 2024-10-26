import requests
import streamlit as st

def get_response(input_text):
    response = requests.post("http://localhost:8000/summary/invoke",
    json = {'input':{'topic':input_text}})

    return response.json()['output']

st.title('Charater summary Ollama')
input_text=st.text_input("Write a basic summary on : ")

if input_text:
    st.write(get_response(input_text))
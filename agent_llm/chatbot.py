import os

from dotenv import load_dotenv
from langchain_together import ChatTogether


load_dotenv()


chatbot = None

def create_chatbot():

    global chatbot

    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    chatbot = ChatTogether(api_key=api_key, model=model_name)

    return chatbot
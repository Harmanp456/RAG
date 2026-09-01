import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = PyPDFLoader("document loaders/GRU.pdf").load()

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarizes the text."),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model="mistral-small-latest")

prompt = template.format_messages(data=data[0].page_content)
result = model.invoke(prompt)

print(result.content)
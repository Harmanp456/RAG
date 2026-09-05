import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf").load()

splitter=   RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split the text into chunks of 1000 characters with 10 characters overlap
chunk=splitter.split_documents(data)
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarizes the text."),
        ("human", "{data}")
    ]
)

model = ChatMistralAI(model="mistral-small-latest")

prompt = template.format_messages(data=chunk[0].page_content)
result = model.invoke(prompt)

print(result.content)
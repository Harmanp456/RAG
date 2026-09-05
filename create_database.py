#load pdf
# split into chunks
#create the embeddings
#store in chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

loader=PyPDFLoader("document loaders/deeplearning.pdf")
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks=splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db",
)


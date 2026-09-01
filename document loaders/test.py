from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

docs = TextLoader("document loaders/notes.txt").load()

print(docs[0].page_content)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter  
data=PyPDFLoader("document loaders/GRU.pdf")
docs=data.load()

splitter=TokenTextSplitter(chunk_size=1000, chunk_overlap=10)  # Split the text into chunks of 1000 tokens with no overlap

chunks=splitter.split_documents(docs)  # Split the documents into chunks

#  har page ka ek doc create hoga if page 15 then there are 15 docs print (len(data))  # print the number of docs
print (len(chunks))

print(chunks[0].page_content)  # print the number of chunks
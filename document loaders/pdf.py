from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("document loaders/GRU.pdf")
docs=data.load()
#  har page ka ek doc create hoga if page 15 then there are 15 docs print (len(data))  # print the number of docs
print (docs[14])
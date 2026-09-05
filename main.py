import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from httpx import HTTPStatusError

from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


embeddings = MistralAIEmbeddings(model="mistral-embed")
vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm = ChatMistralAI(
    model_name="mistral-small-latest",
    temperature=0.2,
    max_tokens=512,
    max_retries=0,
)

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    try:
        response = llm.invoke(final_prompt)
        print(f"\n AI: {response.content}")
    except HTTPStatusError as error:
        if error.response.status_code == 429:
            print("\n Mistral API rate limit reached. Wait for the quota window to reset and try again.")
        else:
            print(f"\n Mistral API error ({error.response.status_code}). Check your API key and account status.")
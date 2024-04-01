PINECONE_API_KEY =''
INDEX_NAME = 'bot'
OPENAI_API_KEY = ''


import openai
# from insertion import pc 
from pinecone import Pinecone
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("bot")
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
INDEX_NAME='bot'

# os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("bot")


def answer_with_gpt(query: str,context = None,):
    messages = [
        {"role" : "system", "content":'''Please give answer from give content'''}
    ]
    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages
        )

    return '\n' + response.choices[0].message.content.strip()

embedding_fn=OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_fn)

prompt = "FY 2023. Our revenue for the year was ?"

# docs = index.query(prompt)
docs=vectorstore.similarity_search(prompt,k=8)
context = ""
for doc in docs:
    context +=f"\n\n{doc.page_content}"
response = answer_with_gpt(prompt,context)
open("context.txt","w",encoding="utf-8").write(context)
print(response)

 
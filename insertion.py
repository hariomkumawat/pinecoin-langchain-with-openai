PINECONE_API_KEY =''
INDEX_NAME = 'bot'
OPENAI_API_KEY = ''


import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter




pc = Pinecone(api_key=PINECONE_API_KEY)

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY


## Lets Read the document
def read_doc (directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents



def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks

loader = read_doc("documents")

docs = chunk_data(docs=loader)




def insert_vectors(docs):
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    print(f"Started data insertion")
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print(f"INFO: Vectors created successfully on index: {INDEX_NAME}")


insert_vectors(docs)

# index_name = "bot"


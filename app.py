import os

import openai
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from flask import render_template,request,Flask,redirect
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("bot")
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
INDEX_NAME='bot'

@app.route('/')
def main():
    return render_template('chat.html')




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

    res= '\n' + response.choices[0].message.content.strip()
    # print(res)
    return res


@app.route('/query', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        query = str(request.args.get('text'))
        print(query)
        embedding_fn=OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_fn)
        docs=vectorstore.similarity_search(query,k=8)
        context = ""
        for doc in docs:
            context +=f"\n\n{doc.page_content}"
        res = answer_with_gpt(query,context)
        return res
    return "Error: Invalid request method"


if __name__ == '__main__': 
	app.run(debug=True)
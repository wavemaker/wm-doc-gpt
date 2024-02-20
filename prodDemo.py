from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import uuid
import chromadb
from flask import Flask, request, jsonify
from uuid import uuid1


chroma_client = chromadb.Client()

loader = DirectoryLoader('../MarketingRag/data/source_data', glob="./*.md", loader_cls=TextLoader)
data = loader.load()

llm = ChatOpenAI(model_name='gpt-4-turbo-preview', 
                 temperature=0)

embeddings = OpenAIEmbeddings()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                          chunk_overlap=100)
chunks = splitter.split_documents(data)

data = {}
for doc in chunks:
    data[str(uuid.uuid1())] = {
        "source": doc.metadata, 
        "documents": doc.page_content
    }
    break

vectorstore = Chroma.from_documents(chunks, 
                                    embeddings,
                                    collection_name="wm1")

vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 5})

keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k =  5

ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   keyword_retriever],
                                                  weights=[0.5, 0.5],
                                                  return_source_documents=True
                                                  )


template = """
<|system|>>
You are an assistant for WaveMaker, drawing from the provided data. If you encounter a question to which you don't know the answer, simply respond with 'Sorry, I'm not familiar with this question please visit the website https://www.wavemaker.com' or refrain from adding any unnecessary information.

CONTEXT: {context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""


prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

chain = (
    {"context": ensemble_retriever, 
     "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

def answer_question(question):
    if question:
        answer = chain.invoke(question)
        return answer
    else:
        return "Ask me anything!"

def sourceData(question):
    result = ensemble_retriever.invoke(question)
    return result

app = Flask(__name__)

@app.route('/finalQan', methods=['POST'])
def finalQan():
    question = request.json.get('question')

    answer = answer_question(question)
    source_data = sourceData(question)
    serializable_source_data = []
    
    for doc in source_data:
        serializable_doc = {
            "page_content": doc.page_content,
            "metadata": {
                "source": doc.metadata["source"]
            }
        }
        serializable_source_data.append(serializable_doc)
    return jsonify({'answer': answer, 'source_data': serializable_source_data})

if __name__ == "__main__":
    app.run(debug=True)
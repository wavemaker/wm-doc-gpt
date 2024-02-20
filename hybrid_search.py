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

chroma_client = chromadb.Client()

loader = DirectoryLoader('/Users/chiranjeevib_500350/wavemaker/Project/MarketingRag/data/source_data', glob="./*.md", loader_cls=TextLoader)
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

iface = gr.ChatInterface(
    answer_question,
    # chatbot=gr.Chatbot(height=),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="Wavemaker Q&A Hub",
    theme="soft",
    # examples=["How is versioning of Applications done on WaveMaker?", "How are workflows managed on WaveMaker?", "How is Unit Testing enabled on WaveMaker?"],
    # cache_examples=True,
    retry_btn=None,
    # undo_btn="Delete Previous",
    # clear_btn="Clear",
    )




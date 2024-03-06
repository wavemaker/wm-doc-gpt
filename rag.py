from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from qdrant_client import models, QdrantClient
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.prepare_db import PrepareVectorDB


from src.config import( 
                    DATA_LOC,
                    HOSTNAME,
                    PORT,
                    QUDRANT_URL,
                    PERSIST_DIRECTORY,
                    COLLECTION_NAME,
                    MODEL,
                    TEMPERATURE,
                    SYSTEM_MSG
                )

embeddings = OpenAIEmbeddings()

qdrant = QdrantClient(HOSTNAME, 
                      port = PORT)

llm = ChatOpenAI(model_name = MODEL, 
                 temperature = TEMPERATURE)


db = Qdrant(client = qdrant, 
            embeddings = embeddings, 
            collection_name = COLLECTION_NAME)

vectorstore_retreiver = db.as_retriever(search_kwargs={"k": 5})


read_docs = PrepareVectorDB(DATA_LOC)
data = read_docs.load_data()
chunks = read_docs.chunk_documents()


keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k =  5

ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                   keyword_retriever],
                                                  weights=[0.5, 0.5],
                                                  return_source_documents=True
                                                  )

prompt = ChatPromptTemplate.from_template(SYSTEM_MSG)
output_parser = StrOutputParser()

chain = (
    {"context": ensemble_retriever, 
     "query": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)



    

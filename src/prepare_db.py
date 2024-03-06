from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import models, QdrantClient
from src.loader import CustomFileLoader, CustomDirectoryLoader
import os
from dotenv import load_dotenv

from src.config import( 
                    DATA_LOC,
                    HOSTNAME,
                    PORT,
                    QUDRANT_URL,
                    PERSIST_DIRECTORY,
                    COLLECTION_NAME
                )

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings() 
qdrant = QdrantClient(HOSTNAME, port=PORT)

import logging
logging.basicConfig(filename='info.log', level=logging.INFO)


class PrepareVectorDB:
    def __init__(self,PATH):
        self.data = None
        self.PATH = PATH
        # self.persist_directory = PERSIST_DIRECTORY

    def load_data(self):
        try:
            if self.PATH.endswith('.json'):
                loader = CustomFileLoader(self.PATH)
                self.data = loader.load()
                
                logging.info("Loading json documents is done!")
                return self.data

            elif self.PATH.endswith('.csv'):
                loader = CustomFileLoader(self.PATH)
                self.data = loader.load()
                logging.info("Loading .csv documents is done")
                return self.data

            else:
                loader = CustomDirectoryLoader(self.PATH, glob="./*.md", loader_cls=TextLoader)
                self.data = loader.load()
                logging.info("Loading .md documents is done")
                return self.data

        except FileNotFoundError as e:
            logging.error(f"File or directory not found: {self.PATH}")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            
    def chunk_documents(self):
        try:
            if self.data is None:

                logging.error("Data is not loaded. Aborting chunk_documents.")
                
                return None

            logging.info("Loading documents for chunking")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                      chunk_overlap=100)
            chunks = splitter.split_documents(self.data)

            logging.info("Chunking of the Data is Done")

            return chunks
        
        except Exception as e:
            logging.error(f"Error chunking documents: {e}")
            return None

    def prepare_and_save_vectordb(self,COLLECTION_NAME):
        self.load_data()

        if self.data is None:
            logging.error("Data is not loaded. Aborting prepare_and_save_vectordb.")
            return None

        chunked_documents = self.chunk_documents()
        if chunked_documents is None:
            logging.error("Chunked documents are not available. Aborting prepare_and_save_vectordb.")
            return None

        try:
            logging.info("Preparing vectordb...")
            vectordb = Qdrant.from_documents(documents=chunked_documents,
                                             embedding=embedding,
                                             url=QUDRANT_URL,
                                             collection_name=COLLECTION_NAME,
                                             prefer_grpc=False)
            logging.info(f"VectorDB is created and saved and stored the embeddings in the vector DB  with collection name {COLLECTION_NAME}")
            return vectordb
        
        except Exception as e:
            logging.error(f"Error preparing and saving vectordb: {e}")
            return None
        
# db = PrepareVectorDB(DATA_LOC)
# res = db.prepare_and_save_vectordb()
# print(res)
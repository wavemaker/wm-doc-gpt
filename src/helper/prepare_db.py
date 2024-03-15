import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import models, QdrantClient
from src.helper.loader import CustomFileLoader, CustomDirectoryLoader
from src.helper.model import SentenceTransformerLoader 
from src.config.config import(
                    HOSTNAME,
                    PORT,
                    QUDRANT_URL,
                    PERSIST_DIRECTORY,
                    QDRANT_API_KEY
                )

# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings() 
qdrant = QdrantClient(HOSTNAME, 
                      port=PORT,
                      #api_key=QDRANT_API_KEY
                      )
encoder = SentenceTransformerLoader.get_model()

class PrepareVectorDB:
    def __init__(self,PATH):
        self.data = None
        self.PATH = PATH
        self.logger = logging.getLogger(__name__)
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
                loader = CustomDirectoryLoader(self.PATH, 
                                               glob="**/*.md", 
                                               loader_cls=TextLoader)
                self.data = loader.load()
                if self.data != None:
                    logging.info("Loading .md documents is done")
                    return self.data
                else:
                    return None
                

        except FileNotFoundError as e:
            logging.error(f"File or directory not found: {self.PATH}")
            return None

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None
            
    def chunk_documents(self):
        try:
            if self.data is None:
                logging.error("Data is not loaded. Aborting chunk_documents.")
                return None

            logging.info("Loading documents for chunking")

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, 
                                                      chunk_overlap=50)
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
        
class SemanticSearch:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.encoder = encoder
        self.logger = logging.getLogger(__name__)

    def create_collection(self):
        try:
            vectors_config = models.VectorParams(
                            size=self.encoder.get_sentence_embedding_dimension(),
                            distance=models.Distance.COSINE
                            )
            
            qdrant.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=vectors_config
                            )
            self.logger.info(f"Successfully created collection with collection name as:{self.collection_name}.")

        except Exception as e:
            self.logger.error(f"Error occurred: {e}")

    def upload_to_collection(self, faqdata):
        try:
            points = [
                models.PointStruct(
                    id=idx,
                    vector=self.encoder.encode(doc["answer"]).tolist(),
                    payload=doc
                ) for idx, doc in enumerate(faqdata)
            ]
            qdrant.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            self.logger.info(f"Successfully uploaded data to collection for {self.collection_name}.")
        
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
        
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
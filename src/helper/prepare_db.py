import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import VectorParams, Distance 
from src.helper.loader import CustomFileLoader, CustomDirectoryLoader
from src.helper.model import SentenceTransformerLoader 
from src.config.config import(
                    HOSTNAME,
                    PORT,
                    QUDRANT_URL,
                    PERSIST_DIRECTORY,
                    QDRANT_API_KEY,
                    CUSTOM_QDRANT_CLIENT,
                    COLLECTION_NAME,
                    WAVEMAKER_WEBSITE,
                    FAQ_COLLECTION_NAME
                )

# load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings() 

encoder = SentenceTransformerLoader.get_model()
qdrant_scraper_client = Qdrant(CUSTOM_QDRANT_CLIENT, 
                               COLLECTION_NAME, 
                               embedding)


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

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                      chunk_overlap=100)
            chunks = splitter.split_documents(self.data)

            logging.info("Chunking of the Data is Done")
            return chunks
        
        except Exception as e:
            logging.error(f"Error chunking documents: {e}")
            return None

    def prepare_and_save_vectordb(self):
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

class PrepareAndSaveScrappedData():
    def __init__(self,PATH):
        self.PATH = PATH
        self.scrapped_data = None
        self.logger = logging.getLogger(__name__)

    def load_scrapped_data(self):
        try:
            data_loader = CustomDirectoryLoader(self.PATH, 
                                        glob="**/*.md", 
                                        loader_cls=TextLoader)
            self.scrapped_data = data_loader.load()
            
            if self.scrapped_data is not None:
                logging.info("Loading .md of scraped data is done")
                return self.scrapped_data
            
            else:
                return None
        except Exception as e:
            logging.error(f"Error loading scrapped data: {e}")
            return None
    
    def chunk_scrapped_data(self):
        try:
            if self.scrapped_data is None:
                logging.error("Scrapped data is not loaded. Aborting chunk_scrapped_data.")
                return None
            
            logging.info("Loading scrapped data for chunking")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                      chunk_overlap=100)
            data_chunks = splitter.split_documents(self.scrapped_data)

            logging.info("Chunking of the scrapped data is done")
            return data_chunks
        
        except Exception as e:
            logging.error(f"Error chunking scrapped data: {e}")
            return None
    
    def check_and_create_collection(self, collection_name):
        try:
            collections = CUSTOM_QDRANT_CLIENT.get_collections().collections
            existing_collection = False

            for collection in collections:
                if collection_name == collection.name:
                    existing_collection = True
                    break

            if not existing_collection:
                CUSTOM_QDRANT_CLIENT.recreate_collection(
                    collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                    ),
                )

            return existing_collection
        
        except Exception as e:
            logging.error(f"An error occurred in check_and_create_collection: {str(e)}")


    def prepare_and_save_scrapped_data(self):
        try:
            self.load_scrapped_data()

            if self.scrapped_data is None:
                logging.error("Data is not loaded. Aborting prepare_and_save_scrapped_data.")
                return None
            
            scrapped_data_chunks = self.chunk_scrapped_data()
            
            if scrapped_data_chunks is None:
                logging.error("Chunked documents are not available. Aborting prepare_and_save_scrapped_data.")
                return None

            content_list = []
            source_list = []

            for document in scrapped_data_chunks:
                content_list.append( document.page_content)
                base_filename = os.path.basename(document.metadata["source"])
                full_source_path = os.path.join(WAVEMAKER_WEBSITE, base_filename)
                source_list.append({"source": full_source_path})

            self.check_and_create_collection(COLLECTION_NAME)
            qdrant_scraper_client.add_texts(content_list, source_list)
            logging.info("Data added to db using the add_texts method")

            content_list.clear()
            source_list.clear()
            
            return True
            
        except Exception as e:
            logging.error(f"An error occurred in prepare_and_save_scrapped_data: {str(e)}")

        
class SemanticSearch:
    def __init__(self):
        self.encoder = encoder
        self.logger = logging.getLogger(__name__)

    def create_collection(self):
        try:
            vectors_config = models.VectorParams(
                            size=self.encoder.get_sentence_embedding_dimension(),
                            distance=models.Distance.COSINE
                            )
            
            CUSTOM_QDRANT_CLIENT.create_collection(
                            collection_name=FAQ_COLLECTION_NAME,
                            vectors_config=vectors_config
                            )
            self.logger.info(f"Successfully created collection with collection name as:{FAQ_COLLECTION_NAME}.")

        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
    
    def upload_and_update_collection(self, faqdata):
        try:
            points = [
                models.PointStruct(
                    id=doc["id"],
                    vector=self.encoder.encode(doc["question"]).tolist(),
                    payload=doc
                ) for doc in faqdata
            ]
            CUSTOM_QDRANT_CLIENT.upload_points(
                collection_name=FAQ_COLLECTION_NAME,
                points=points
            )
            self.logger.info(f"Successfully uploaded data to collection for {FAQ_COLLECTION_NAME}.")
        except Exception as e:
            self.logger.error(f"Failed to upload data to collection: {e}")

    def delete_qan(self, id_val):
        try:
            CUSTOM_QDRANT_CLIENT.delete(
                    collection_name=FAQ_COLLECTION_NAME,
                    points_selector=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="id",
                                    match=models.MatchValue(value=int(id_val)),
                                ),
                            ],
                        )
                )
            self.logger.info("Successfully deleted data from the collection.")
            return True  
        except Exception as e:
            self.logger.error(f"Failed to delete data from the collection: {e}")
            return False  

class DeleteDuplicates:
    def __init__(self, url, collection_name):
        self.url = url
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

    def get_id_from_source(self, data, source):
        try:
            ids = []
            for record in data:
                if record.payload['metadata']['source'] == source:
                    ids.append(record.id)
            
            if not ids:
                return None
            return ids
        
        except Exception as e:
            self.logger.error(f"Error getting IDs from source: {e}")
            return None

    
    def delete_vector(self, ids):
        try:
            if not isinstance(ids, (list, tuple)):
                ids = [ids]  
            
            for id in ids:
                CUSTOM_QDRANT_CLIENT.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=[id],
                    ),
                )
            self.logger.info("Vectors deleted successfully.")
        
        except Exception as e:
            self.logger.error(f"Error deleting vectors: {e}")


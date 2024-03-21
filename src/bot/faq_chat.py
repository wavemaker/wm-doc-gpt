from qdrant_client import models, QdrantClient
from src.helper.loader import CustomFileLoader
from src.helper.prepare_db import SemanticSearch
from src.helper.model import SentenceTransformerLoader
import logging

from src.config.config import (
                                HOSTNAME,
                                PORT,
                                FAQ_LOC,
                                FAQ_COLLECTION_NAME,
                                QDRANT_API_KEY
                            )

qdrantClient = QdrantClient(HOSTNAME, 
                            port = PORT,
                            #api_key=QDRANT_API_KEY
                            )

encoder = SentenceTransformerLoader.get_model()

class CollectionUploadChecker:
    def __init__(self, faq_collection_name, FAQ_LOC):
        self.faq_collection_name = faq_collection_name
        self.FAQ_LOC = FAQ_LOC
        self.qdrantClient = qdrantClient

    def check_collection_upload_data(self):
        collections = self.qdrantClient.get_collections().collections
        existing_collection = False
        
        for collection in collections:
            list_collection = collection.name
            if self.faq_collection_name == list_collection:
                existing_collection = True
                break

        try:        
            read_data = CustomFileLoader(self.FAQ_LOC)
            data = read_data.load()

            if data != None:
                if existing_collection:
                    res = SemanticSearch(self.faq_collection_name)
                    res.upload_to_collection(data)
                    logging.info(f"FAQ data uploaded successfully to the existing collection! {self.faq_collection_name}")
                    return "FAQ data uploaded successfully to the existing collection"
                else:
                    res = SemanticSearch(self.faq_collection_name)
                    res.create_collection()
                    res.upload_to_collection(data)
                    logging.info(f"FAQ data uploaded successfully! {self.faq_collection_name}")
                    return "FAQ data uploaded successfully"
            else:
                return None
        except Exception as e:
            logging.error(f"Data not location not found: {e}")
            return None

def search(question):
        hits = qdrantClient.search(
                                collection_name=FAQ_COLLECTION_NAME,
                                query_vector=encoder.encode(question).tolist(),
                                limit=3,
                                )
        return hits

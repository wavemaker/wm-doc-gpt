from qdrant_client import models, QdrantClient
from src.helper.loader import CustomFileLoader
from src.helper.prepare_db import SemanticSearch
from src.helper.model import SentenceTransformerLoader
import logging

from src.config.config import (
                                FAQ_LOC,
                                FAQ_COLLECTION_NAME,
                                QDRANT_API_KEY,
                                CUSTOM_QDRANT_CLIENT
                            )

encoder = SentenceTransformerLoader.get_model()

class CollectionUploadChecker:
    def __init__(self, faq_collection_name, faq_data_loc):
        self.faq_collection_name = faq_collection_name
        self.faq_data_loc = faq_data_loc
        self.CUSTOM_QDRANT_CLIENT = CUSTOM_QDRANT_CLIENT

    def check_collection_upload_data(self):
        collections = self.CUSTOM_QDRANT_CLIENT.get_collections().collections
        existing_collection = False
        
        for collection in collections:
            list_collection = collection.name
            
            if self.faq_collection_name == list_collection:
                existing_collection = True
                break

        try:        
            read_data = CustomFileLoader(self.faq_data_loc)
            data = read_data.load()

            if data is not None:
                if existing_collection:
                    res = SemanticSearch()
                    if res.upload_and_update_collection(data):
                        message = f"FAQ data uploaded successfully to the existing collection! {self.faq_collection_name}"
                        logging.info(message)
                        return True, message
                    else:
                        message = "Failed to upload FAQ data to the existing collection"
                        logging.error(message)
                        return False
                else:
                    res = SemanticSearch()
                    if res.create_collection():
                        if res.upload_and_update_collection(data):
                            message = f"FAQ data uploaded successfully! {self.faq_collection_name}"
                            logging.info(message)
                            return True
                        else:
                            message = "Failed to upload FAQ data"
                            logging.error(message)
                            return False, message
                    else:
                        message = "Failed to create collection"
                        logging.error(message)
                        return False
            else:
                return False
        except Exception as e:
            logging.error(f"Data location not found: {e}")
            return False


def search(question):
        hits = CUSTOM_QDRANT_CLIENT.search(
                                collection_name=FAQ_COLLECTION_NAME,
                                query_vector=encoder.encode(question).tolist(),
                                limit=3,
                                )
        return hits

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

            if data != None:
                if existing_collection:
                    res = SemanticSearch()
                    res.upload_and_update_collection(data)
                    logging.info(f"FAQ data uploaded successfully to the existing collection! {self.faq_collection_name}")
                    return "FAQ data uploaded successfully to the existing collection"
                else:
                    res = SemanticSearch()
                    res.create_collection()
                    res.upload_and_update_collection(data)
                    logging.info(f"FAQ data uploaded successfully! {self.faq_collection_name}")
                    return "FAQ data uploaded successfully"
            else:
                return None
        except Exception as e:
            logging.error(f"Data not location not found: {e}")
            return None


def search(question):
        hits = CUSTOM_QDRANT_CLIENT.search(
                                collection_name=FAQ_COLLECTION_NAME,
                                query_vector=encoder.encode(question).tolist(),
                                limit=3,
                                )
        return hits

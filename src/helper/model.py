from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDING_MODEL

class SentenceTransformerLoader:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            cls._instance = SentenceTransformer(EMBEDDING_MODEL)
        return cls._instance



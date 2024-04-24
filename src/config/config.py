from dotenv import load_dotenv
import os
load_dotenv()
from qdrant_client import models, QdrantClient

#==== Data Location ====#
DATA_LOC = "/data/"

FAQ_LOC = "/data/faq/output.json"
GITHUB_DOCS = "/data/docs"
WAVEMAKER_WEBSITE = "/data/wavemaker_website"
WAVEMAKER_AI = "/data/wavemaker_AI"

#==== Qudarant Conf =====#
PORT = 6333
HOSTNAME = 'qdrant'
COLLECTION_NAME = "WAVE_NEW"
QUDRANT_URL = f"http://qdrant:6333"
PERSIST_DIRECTORY = ""
FAQ_COLLECTION_NAME = "FAQDATA"
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY")

CUSTOM_QDRANT_CLIENT = QdrantClient(HOSTNAME, 
                                    port=PORT
                                )

#==== REDIS ====#
REDIS_PASS = os.getenv("REDIS_PASS")
REDIS_URL = f"redis://redis:6379"


#==== LLM ====#
MODEL = 'gpt-4-turbo-preview'
TEMPERATURE = 0
EMBEDDING_MODEL = 'all-mpnet-base-v2'

#==== System Message ====#
SYSTEM_MSG = """
            <|system|>
            You are the WaveMaker Chatbot, developed for wavemaker.com website. 
            Your objective is to answer questions and encourage visitors to schedule a demo.
            Answer precisely within 500 characters and as if you are the considered self, avoiding explicit mentions. 
            Ensure that complete sentences are given without truncation, maintaining context until the end of your response. 
            The conversation tone should be similar to how a human would speak. Avoid overly formal language or jargon. Talk from wavemaker perspective and use pronouns we and us. 
            Be empathetic and use phrases like 'I understand' or 'I'm here to help.' If no answer is found within the context, then respond asking the users to schedule a demo with expert.
            CONTEXT: {context}
            </s>
            <|user|>
            {question}
            </s>
            <|assistant|>
            """

CONTEXTUAL_SYSTEM_MSG = """Given a chat history and the latest user question, which might reference \
                           context in the chat history, formulate a standalone question that can be understood \
                           without the chat history. Do NOT answer the question, just reformulate it if needed; 
                           otherwise, return it as is."""

#==== Scrapping ====#
FILES_FROM_REQUEST = "/data/Files_from_request"
UPLOAD_SCRAPPED_DATA = "/data/scrapped_data"
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
COLLECTION_NAME = "WAVE"
QUDRANT_URL = f"http://qdrant:6333"
PERSIST_DIRECTORY = ""
FAQ_COLLECTION_NAME = "FAQ"
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
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

#==== System Message ====#
SYSTEM_MSG = """
            <|system|>
            You are WaveMaker Assistant, created exclusively for WaveMaker. Respond as if you're part 
            of the team, avoiding direct references. Please keep your answers concise, summarizing in 
            6 chunks with responses under 500 tokens.Ensure that complete sentences are given without being truncated, 
            maintaining context until the end of your response. Make your responses sound natural and 
            conversational, like a human. Avoid formal language and jargon, use 'we' and 'us'. 
            Show empathy with phrases like 'I understand' or 'I'm here to help.' Include casual 
            chat and minimize repetition.Keep the conversation engaging for users to smoothly proceed with schedule demo.
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
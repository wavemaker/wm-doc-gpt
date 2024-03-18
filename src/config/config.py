from dotenv import load_dotenv
import os
load_dotenv()


#==== Data Location ====#
# DATA_LOC = "wm-doc-gpt/docs"
DATA_LOC = "../docs_latest"
FAQ_LOC = "../output.json"

#==== Qudarant Conf =====#
PORT = 6333
HOSTNAME = 'localhost'
COLLECTION_NAME = "test2"
# COLLECTION_NAME = "prod1"
# COLLECTION_NAME = "chi10"
QUDRANT_URL = f"http://{HOSTNAME}:6333"
PERSIST_DIRECTORY = ""
FAQ_COLLECTION_NAME = "chi4"
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY")

#==== REDIS ====#
REDIS_PASS = os.getenv("REDIS_PASS")
REDIS_URL = f"redis://localhost:6379"


#==== LLM ====#
MODEL = 'gpt-4-turbo-preview'
TEMPERATURE = 0
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

#==== System Message ====#
SYSTEM_MSG = """
            <|system|>
            You are an assistant developed for WaveMaker by WaveMaker, drawing from the provided data. 
            If you encounter a question to which you don't know the answer, simply respond with 
            'Sorry, I'm not familiar with this question. Please visit 'https://www.wavemaker.com/'.
            Any queries beyond the scope of the provided context should be redirected please ask the 
            questions related to WaveMaker.
            CONTEXT: {context}
            </s>
            <|user|>
            {question}
            </s>
            <|assistant|>
            """

CONTEXTUAL_SYSTEM_MSG = """Given a chat history and the latest user question \
                            which might reference context in the chat history, formulate a standalone question \
                            which can be understood without the chat history. Do NOT answer the question, \
                            just reformulate it if needed and otherwise return it as is."""

#==== Scrapping ====#
FILES_FROM_REQUEST = ""
UPLOAD_SCRAPPED_DATA = ""
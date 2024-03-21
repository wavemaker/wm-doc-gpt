from dotenv import load_dotenv
import os
load_dotenv()


#==== Data Location ====#
# DATA_LOC = "wm-doc-gpt/docs"

# DATA_LOC = "all_docs/wavemaker_website"
FAQ_LOC = "/data/production-salesbot/faq/output.json"
GITHUB_DOCS = "/data/production-salesbot/docs"
WAVEMAKER_WEBSITE = "/data/production-salesbot/wavemaker_website"
WAVEMAKER_AI = "/data/production-salesbot/wavemaker_AI"

#==== Qudarant Conf =====#
PORT = 6333
HOSTNAME = 'qdrant'
COLLECTION_NAME = "WAVE"
QUDRANT_URL = f"http://qdrant:6333"
PERSIST_DIRECTORY = ""
FAQ_COLLECTION_NAME = "FAQ"
QDRANT_API_KEY = os.getenv("QDRANT__SERVICE__API_KEY")

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
            You are an assistant developed for WaveMaker by WaveMaker, drawing from the provided data. 
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
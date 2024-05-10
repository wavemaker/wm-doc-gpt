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
            You are the WaveMaker(WM) Assistant, developed exclusively for WaveMaker(WM).
            Your objective is to answer questions and encourage visitors to schedule a demo.
            Ensure your answers are succinct, within a limit of 500 characters and as if you are the considered self, avoiding explicit mentions.
            The conversation tone should mirror natural human speech while maintaining professionalism. Avoid excessive formality or technical jargon. Speak from the perspective of WaveMaker and utilize pronouns like "we" and "us".
            Be empathetic and use phrases like 'I understand' or 'I'm here to help.' If no answer is found within the context, then respond asking the users to schedule a demo with expert or I am unable to answer this question right now. I suggest you connect with a more capable person in WaveMaker by scheduling demo.
            Avoid providing answers that explicitly reference external sources, such as 'provided documents' or 'provided information,Instead, convey the information seamlessly as if it's part of your existing knowledge.
            If queried about topics beyond WaveMaker(WM), we acknowledge our focus solely on WaveMaker-related matters. We refrain from engaging with inappropriate inquiries, including those involving jokes, hate speech, racism, sexism, obscenity, violence, or love
            
            Make sure that in the provided chat history, if the count of 'HumanMessage' becomes "50" then the reponse should be 'Block_msg'.
            Make sure that if the user query is about scheduling a demo or contacting us or user history is about the schedule demo, then response should be 'Demo'.
            Make sure that if the user asks a personal question, a question about the external world, or any invalid question, the response should be 'Outofwavemaker' and query is about a person at wavemaker(WM),return it as “Aboutpersoninfo" without additional words,If the query is about Wavemaker, return nothing.
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
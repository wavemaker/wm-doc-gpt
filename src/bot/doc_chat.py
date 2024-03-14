from langchain.vectorstores import Qdrant
from qdrant_client import models, QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.helper.prepare_db import PrepareVectorDB
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging

from src.config.config import( 
                    DATA_LOC,
                    HOSTNAME,
                    PORT,
                    COLLECTION_NAME,
                    MODEL,
                    TEMPERATURE,
                    SYSTEM_MSG,
                    CONTEXTUAL_SYSTEM_MSG,
                )

class ChatAssistant:
    loaded_chunks = None
    ensemble_retriever = None 
    keyword_retriever = None
    
    @classmethod
    def load_chunks(cls, DATA_LOC):
        if cls.loaded_chunks is None:
            read_docs = PrepareVectorDB(DATA_LOC)
            data = read_docs.load_data()
            cls.loaded_chunks = read_docs.chunk_documents()  

    @staticmethod
    def rag(session_id, question, url):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        embeddings = OpenAIEmbeddings()
        qdrant = QdrantClient(HOSTNAME, port=PORT)
        llm = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)
        db = Qdrant(
            client=qdrant, embeddings=embeddings, collection_name=COLLECTION_NAME)
        
        vectorstore_retriever = db.as_retriever(search_kwargs={"k": 2})
        
        if ChatAssistant.loaded_chunks is None:
            ChatAssistant.load_chunks(DATA_LOC)

        if ChatAssistant.keyword_retriever is None:
            ChatAssistant.keyword_retriever = BM25Retriever.from_documents(ChatAssistant.loaded_chunks)
            ChatAssistant.keyword_retriever.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever, ChatAssistant.keyword_retriever],
            weights=[0.6, 0.4],
            return_source_documents=True)
    
        ChatAssistant.ensemble_retriever = ensemble_retriever

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MSG),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUAL_SYSTEM_MSG),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        @staticmethod
        def contextualized_question(input_dict):
            if input_dict.get("chat_history"):
                return contextualize_q_chain
            else:
                return input_dict["question"]

        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | ensemble_retriever
            )
            | qa_prompt
            | llm
        )

        @staticmethod
        def get_message_history(session_id: str) -> RedisChatMessageHistory:
            return RedisChatMessageHistory(session_id, url=url)

        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            get_message_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

        return with_message_history

    @staticmethod
    def answer_question(session_id, question,url):
        history = RedisChatMessageHistory(session_id, url=url)
        with_message_history = ChatAssistant.rag(session_id, question,url)
        if question:
            answer = str(with_message_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                    ))
            history.add_user_message(question)
            history.add_ai_message(answer)
            return answer
        else:
            return "Ask me anything!"

    @staticmethod
    def get_sources(question):
        if ChatAssistant.ensemble_retriever is None:
            raise ValueError("Ensemble retriever not initialized")
        
        source_url = ChatAssistant.ensemble_retriever.invoke(question)
        return source_url

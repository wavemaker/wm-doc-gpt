from langchain.vectorstores import Qdrant
from qdrant_client import models, QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.helper.prepare_db import PrepareVectorDB
import logging
from flask import jsonify
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
    keyword_retriever = None
    
    @classmethod
    def load_chunks(cls, DATA_LOC):
        if cls.loaded_chunks is None:
            read_docs = PrepareVectorDB(DATA_LOC)
            data = read_docs.load_data()
            cls.loaded_chunks = read_docs.chunk_documents()  

    @staticmethod
    def rag(session_id, question, url):
        embeddings = OpenAIEmbeddings()
        qdrant = QdrantClient(
                            HOSTNAME, 
                            port=PORT
                            )
        
        llm = ChatOpenAI(
                        model_name=MODEL, 
                        temperature=TEMPERATURE
                         )
        
        db = Qdrant(
                    client=qdrant, 
                    embeddings=embeddings, 
                    collection_name=COLLECTION_NAME
                    )
        
        vectorstore_retriever = db.as_retriever(search_kwargs={"k": 5})
        
        if ChatAssistant.loaded_chunks is None:
            ChatAssistant.load_chunks(DATA_LOC)

        if ChatAssistant.keyword_retriever is None:
            ChatAssistant.keyword_retriever = BM25Retriever.from_documents(ChatAssistant.loaded_chunks)
            ChatAssistant.keyword_retriever.k = 5
        
        ChatAssistant.keyword_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(
                        retrievers=[vectorstore_retriever, 
                                    ChatAssistant.keyword_retriever
                                    ],
                        weights=[0.5, 0.5],
                        return_source_documents=True
                        )

        ChatAssistant.ensemble_retriever = ensemble_retriever

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MSG),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUAL_SYSTEM_MSG),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_chain = contextualize_prompt | llm | StrOutputParser()

        @staticmethod
        def contextualized_question(input_dict):
            if input_dict.get("chat_history"):
                return contextualize_chain
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
        history = RedisChatMessageHistory(session_id, 
                                          url=url)
        with_message_history = ChatAssistant.rag(session_id, 
                                                 question,url)
        docs = ChatAssistant.ensemble_retriever.invoke(question)
        sources = [doc.metadata['source'] for doc in docs]

        def add_website_url(file_path):
            website_url = 'https://docs.wavemaker.com'
            if 'learn' in file_path:
                trimmed_path = file_path.replace('.md', '')

                result_url = f"{website_url}/learn{trimmed_path.split('learn')[1]}"
                return result_url
            elif 'blog' in file_path:
                date_parts = file_path.split('/')[-1].split('-')
                formatted_date = '/'.join(date_parts[:3]) if len(date_parts) >= 3 else ""

                title_parts = file_path.split('/')[-1].split('-')
                title = '-'.join(title_parts[3:]) if len(title_parts) >= 4 else ""

                # Remove .md extension from title
                if title.endswith('.md'):
                    title = title[:-3]

                result_url = f"{website_url}/learn/blog/{formatted_date}/{title}/"
                return result_url
            else:
                return 

        sou = []
        for doc in docs:
            # print(doc.metadata['source'])
            sou.append(add_website_url(doc.metadata['source']))

        if question:
            answer = str(with_message_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                    ))
            answer = answer.replace("content=", "")
            history.add_user_message(question)
            history.add_ai_message(answer)
            return jsonify({'ragAnswer': answer, 
                            'sources':sou})
        else:
            return "Ask me anything about wavemaker!"



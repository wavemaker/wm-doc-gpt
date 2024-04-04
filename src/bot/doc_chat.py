from langchain.vectorstores import Qdrant
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
from flask import jsonify
from src.config.config import( 
                    DATA_LOC, 
                    COLLECTION_NAME,
                    MODEL,
                    TEMPERATURE,
                    SYSTEM_MSG,
                    CONTEXTUAL_SYSTEM_MSG,
                    CUSTOM_QDRANT_CLIENT
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
        
        llm = ChatOpenAI(
                        model_name=MODEL, 
                        temperature=TEMPERATURE
                         )
        
        db = Qdrant(
                    client=CUSTOM_QDRANT_CLIENT, 
                    embeddings=embeddings, 
                    collection_name=COLLECTION_NAME
                    )
        
        vectorstore_retriever = db.as_retriever(search_kwargs={"k": 3})
        
        if ChatAssistant.loaded_chunks is None:
            ChatAssistant.load_chunks(DATA_LOC)

        if ChatAssistant.keyword_retriever is None:
            ChatAssistant.keyword_retriever = BM25Retriever.from_documents(ChatAssistant.loaded_chunks)
            ChatAssistant.keyword_retriever.k = 3
        
        ChatAssistant.keyword_retriever.k = 3

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
                ("human", "{question}")
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
        
        # print(docs)
        # sources = [doc.metadata['source'] for doc in docs]

        def add_website_url(file_path):
            docs_url = 'https://docs.wavemaker.com'
            
            if 'learn' in file_path and 'docs' in file_path:
                trimmed_path = file_path.replace('.md', '')

                result_url = f"{docs_url}/learn{trimmed_path.split('learn')[1]}"
                return result_url
            
            elif 'blog' in file_path and 'docs' in file_path:
                date_parts = file_path.split('/')[-1].split('-')
                formatted_date = '/'.join(date_parts[:3]) if len(date_parts) >= 3 else ""

                title_parts = file_path.split('/')[-1].split('-')
                title = '-'.join(title_parts[3:]) if len(title_parts) >= 4 else ""

                if title.endswith('.md'):
                    title = title[:-3]

                result_url = f"{docs_url}/learn/blog/{formatted_date}/{title}/"
                return result_url
            # elif "wavemaker_website" and "wp-content-uploads" in file_path:
            #     file_path_without_extension = file_path.replace('.md', '')
            #     filename = filename.replace("-", "/")
            #     filename = "https://" + filename
            #     return filename

            elif "wavemaker_website" in file_path and "wp-content-uploads" in file_path:
                file_path_without_extension = file_path.replace('.md', '')
                filename = file_path_without_extension.split('/')[-1] 
                filename = filename.replace("-", "/")
                filename = "https://" + filename  
                return filename

            elif 'wavemaker_website' in file_path:
                file_path_without_extension = file_path.replace('.md', '')
                
                website_url = 'https://www.wavemaker.com'
                url = website_url + file_path_without_extension.split('/wavemaker_website')[1]
                return url
            
            elif "wavemaker_AI" in file_path:
                file_path_without_extension = file_path.replace('.md', '')

                websiteAI_url = 'https://wavemaker.ai'
                waveai_url = websiteAI_url + file_path_without_extension.split('/wavemaker_AI')[1]
                return waveai_url
            else:
                file_path_without_extension = file_path.replace('.md', '')
                return file_path_without_extension

        sources_with_link = []

        for doc in docs:
            sources_with_link.append(add_website_url(doc.metadata['source']))

        unique_sources_with_link = list(set(sources_with_link))

        if question:
            answer = str(with_message_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                    ))
            
            if answer.startswith("content="):
                answer = answer[len("content="):]
                content_without_quotes = answer.replace("'", "")
                
            history.add_user_message(question)
            history.add_ai_message(answer)
            
            if "Question type:" in answer:
                index = answer.find("Question type:")
                output_string = answer[:index].strip()
                return jsonify({'ragAnswer': output_string})
            else:
                return jsonify({'ragAnswer': content_without_quotes, 
                            'sources':unique_sources_with_link})
        else:
            return "Ask me anything about wavemaker!"
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
from langchain_together import ChatTogether
import json
import os
from flask import jsonify
import logging
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

        model_choice = os.getenv('MODEL_CHOICE')
        
        if model_choice == 'Llama':
            logging.info("Llama is being used")

            llm = ChatTogether(
                together_api_key=os.getenv('TOGETHER_API'),
                model="meta-llama/Llama-3-8b-chat-hf",
                max_tokens=500
            )
        else :
            logging.info("OpenAI is being used")

            llm = ChatOpenAI(
                model_name=MODEL,  
                temperature=TEMPERATURE,
                max_tokens=500
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
                                    weights=[0.6, 0.4],
                                    return_source_documents=True
                        )

        ChatAssistant.ensemble_retriever = ensemble_retriever
        ChatAssistant.vectorstore_retriever = vectorstore_retriever

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
                                          url=url,
                                          ttl=180)
        with_message_history = ChatAssistant.rag(session_id, 
                                                 question,url)
        ##Sources
        # docs = ChatAssistant.ensemble_retriever.invoke(question)
        docs = ChatAssistant.vectorstore_retriever.invoke(question)
        
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

            # elif "wavemaker_website" in file_path and "wp-content-uploads" in file_path:
            #     file_path_without_extension = file_path.replace('.md', '')
            #     filename = file_path_without_extension.split('/')[-1] 
            #     filename = filename.replace("-", "/")
            #     filename = "https://" + filename  
            #     return filename

            # elif 'wavemaker_website' in file_path:
            #     file_path_without_extension = file_path.replace('.md', '')
                
            #     website_url = 'https://www.wavemaker.com'
            #     url = website_url + file_path_without_extension.split('/wavemaker_website')[1]
            #     return url
            
            # elif "wavemaker_AI" in file_path:
            #     file_path_without_extension = file_path.replace('.md', '')

            #     websiteAI_url = 'https://wavemaker.ai'
            #     waveai_url = websiteAI_url + file_path_without_extension.split('/wavemaker_AI')[1]
            #     return waveai_url
            else:
                file_path_without_extension = file_path.replace('.md', '')
                return file_path_without_extension

        sources_with_link = []

        for doc in docs:
            sources_with_link.append(add_website_url(doc.metadata['source']))

        unique_sources_with_link = list(set(sources_with_link))

        if question:
            answer = with_message_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                    )
                                        
            history.add_user_message(question)
            history.add_ai_message(answer)
            
            # result = answer.content
            
            # if "Question_type" in result:
            #     data = json.loads(result)
            #     question_type = data["Question_type"]
                    
            #     if question_type == "Aboutperson":
            #         return jsonify({"ragAnswer": "For information about the people working at WaveMaker, please check out our website."})
                
            #     elif question_type == "Outofwavemaker":
            #         return jsonify({"ragAnswer": "I'm here to provide information about WaveMaker. If you have any questions or need assistance with our platform, feel free to ask. How can I assist you today?"})

            # else:
            #     return jsonify({"ragAnswer": result, 
            #                     "sources":unique_sources_with_link})

            response_templates = {
                "Block_msg": {
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": "Schedule a demo with us so we can address your questions. Thank you.",
                    "sources": "",
                    "intent": "Block_message"
                },
                "Demo": {
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": "Thank you for your interest in scheduling a demo with us. Kindly provide the following details, and our expert will promptly reach out to you. We are eager to demonstrate how our platform can fulfill your requirements.",
                    "sources": "",
                    "intent": "Demo"
                },
                "Aboutpersoninfo": {
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": "For information about the people working at WaveMaker, please check out our website.",
                    "sources": "",
                    "intent": ""
                },
                "Outofwavemaker": {
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": "Sorry, I'm here to provide information about WaveMaker. If you have any questions or need assistance with our platform, go ahead and ask me.",
                    "sources": "",
                    "intent": "Outofwavemaker"
                }
            }

            answer_content = answer.content
            
            if "demo" in answer_content:
                return jsonify({
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": answer_content,
                    "sources": unique_sources_with_link,
                    "intent": "Demo"
                })
            elif any(phrase in answer_content for phrase in ['contact us', 'reach out', 'contacting us']):
                return jsonify({
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": answer_content,
                    "sources": unique_sources_with_link,
                    "intent": "Contact_us"
                })
                
            elif answer_content in response_templates:
                return jsonify(response_templates[answer_content])
            
            else:
                return jsonify({
                    "response_from": "RAG",
                    "faq_id": "",
                    "question": "",
                    "answer": answer_content,
                    "sources": unique_sources_with_link,
                    "intent": ""
                })
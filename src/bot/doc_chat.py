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
from src.helper.followup_question_gen import FollowUpQuestionGenerator
import json
import os
from flask import jsonify
import logging
from src.config.config import( 
                    DATA_LOC, 
                    DOCS_COLLECTION,
                    WEBSITE_COLLECTION,
                    MODEL,
                    TEMPERATURE,
                    WEBSITE_SYSTEM_MSG,
                    DOCS_SYSTEM_MSG,
                    CONTEXTUAL_SYSTEM_MSG,
                    CUSTOM_QDRANT_CLIENT,
                    WEBSITE_DATA_LOC,
                    DOCS_DATA_LOC,
                    QUESTION_GEN_SYSTEM_MSG,
                    VIDEO_COLLECTION
                )

class ChatAssistant:
    website_loaded_chunks = None
    docs_loaded_chunks = None
    
    website_keyword_retriever = None
    docs_keyword_retriever = None
    transcribe_retriever = None
    
    @classmethod
    def load_chunks(cls, DATA_LOC):
        
        if cls.website_loaded_chunks or cls.docs_loaded_chunks is None:
            read_docs = PrepareVectorDB(DATA_LOC, WEBSITE_COLLECTION)
            data = read_docs.load_data()
            cls.website_loaded_chunks = read_docs.chunk_documents()

            docs = PrepareVectorDB(DATA_LOC, DOCS_COLLECTION)
            docsData = docs.load_data()
            cls.docs_loaded_chunks = docs.chunk_documents()  

    @staticmethod
    def website_rag(session_id, question, url, service_mode):
        embeddings = OpenAIEmbeddings()

        model_choice = os.getenv('MODEL_CHOICE') 

        if model_choice == 'Llama':
            logging.info("Llama is being used")

            llm = ChatTogether(
                together_api_key=os.getenv('TOGETHER_API'),
                model="meta-llama/Llama-3-70b-chat-hf",
                max_tokens=500
            )
        else :
            logging.info("OpenAI is being used")

            llm = ChatOpenAI(
                model_name=MODEL,  
                temperature=TEMPERATURE,
                max_tokens=500
            )                        

        vectorstore_retriever, transcribe_retriever = (
                                                Qdrant(
                                                    client=CUSTOM_QDRANT_CLIENT, 
                                                    embeddings=embeddings, 
                                                    collection_name=collection).as_retriever(search_kwargs={"score_threshold": 0.7, "k": 2} if collection == VIDEO_COLLECTION else {})
                                                for collection in (WEBSITE_COLLECTION, 
                                                                   VIDEO_COLLECTION)
                                                )
        
        
        if ChatAssistant.website_loaded_chunks is None:
            ChatAssistant.load_chunks(WEBSITE_DATA_LOC)

        if ChatAssistant.website_keyword_retriever is None:
            ChatAssistant.website_keyword_retriever = BM25Retriever.from_documents(ChatAssistant.website_loaded_chunks)
            ChatAssistant.website_keyword_retriever.k = 3
        
        ChatAssistant.website_keyword_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
                        retrievers=[vectorstore_retriever, 
                                    ChatAssistant.website_keyword_retriever
                                    ],
                                    weights=[0.6, 0.4],
                                    return_source_documents=True
                        )
        
        ChatAssistant.ensemble_retriever = ensemble_retriever
        ChatAssistant.vectorstore_retriever = vectorstore_retriever
        ChatAssistant.transcribe_retriever = transcribe_retriever

        if service_mode == "website":
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", WEBSITE_SYSTEM_MSG),
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
    def docs_rag(session_id, question, url, service_mode):
        embeddings = OpenAIEmbeddings()
        model_choice = os.getenv('MODEL_CHOICE') 

        if model_choice == 'Llama':
            logging.info("Llama is being used")

            llm = ChatTogether(
                together_api_key=os.getenv('TOGETHER_API'),
                model="meta-llama/Llama-3-70b-chat-hf",
                max_tokens=500
            )
        else :
            logging.info("OpenAI is being used")

            llm = ChatOpenAI(
                model_name=MODEL,  
                temperature=TEMPERATURE,
                max_tokens=500
            )                        
        
        # docs_db = Qdrant(
        #             client=CUSTOM_QDRANT_CLIENT, 
        #             embeddings=embeddings, 
        #             collection_name=DOCS_COLLECTION
        #             )
        # docs_retriever = docs_db.as_retriever()

        docs_retriever, transcribe_retriever = (
                                                Qdrant(
                                                    client=CUSTOM_QDRANT_CLIENT, 
                                                    embeddings=embeddings, 
                                                    collection_name=collection).as_retriever(search_kwargs={"score_threshold": 0.7, "k": 2} if collection == VIDEO_COLLECTION else {})
                                                for collection in (DOCS_COLLECTION, 
                                                                   VIDEO_COLLECTION)
                                                )

        
        
        if ChatAssistant.docs_loaded_chunks is None:
            ChatAssistant.load_chunks(DOCS_DATA_LOC)

        if ChatAssistant.docs_keyword_retriever is None:
            ChatAssistant.docs_keyword_retriever = BM25Retriever.from_documents(ChatAssistant.docs_loaded_chunks)
            ChatAssistant.docs_keyword_retriever.k = 3
        
        ChatAssistant.docs_keyword_retriever.k = 3
        ensemble_retriever = EnsembleRetriever(
                        retrievers=[docs_retriever, 
                                    ChatAssistant.docs_keyword_retriever
                                    ],
                                    weights=[0.6, 0.4],
                                    return_source_documents=True
                        )
        
        ChatAssistant.ensemble_retriever = ensemble_retriever
        ChatAssistant.docs_retriever = docs_retriever
        ChatAssistant.transcribe_retriever = transcribe_retriever
        
        if service_mode == "docs":
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", DOCS_SYSTEM_MSG),
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
    def answer_question(session_id=None, question=None, url=None, question_from=None):
        history = RedisChatMessageHistory(session_id, 
                                          url=url,
                                          ttl=60)
        if  question_from == "website":
            with_message_history = ChatAssistant.website_rag(session_id, 
                                                    question,
                                                    url,
                                                    question_from)
            
            docs = ChatAssistant.vectorstore_retriever.invoke(question)
            sources = [doc.metadata['source'] for doc in docs]

            videos_ = ChatAssistant.transcribe_retriever.invoke(question)
            dara = [doc.metadata['source'] for doc in videos_]
            dara = list(set(dara))


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
            
                else:
                    file_path_without_extension = file_path.replace('.md', '')
                    return file_path_without_extension

            sources_with_link = []

            for doc in docs:
                sources_with_link.append(add_website_url(doc.metadata['source']))

            unique_sources_with_link = list(set(sources))


            if question:
                answer = with_message_history.invoke(
                        {"question": question},
                        config={"configurable": {"session_id": session_id}},
                        )
                    
                history.add_user_message(question)
                history.add_ai_message(answer.content)
                #---
                

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
                        "intent": "Aboutpersoninfo"
                    },
                    "Outofwavemaker": {
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "answer": "Sorry,I'm here to provide information about WaveMaker. If you have any questions or need assistance with our platform, feel free to ask. How can I assist you today?",
                        "sources": "",
                        "intent": "Outofwavemaker"
                    }
                }
                
                #----
                generator = FollowUpQuestionGenerator()
                follow_up_questions = generator.generate_followup_questions(question)
                follow_up_questions_string = follow_up_questions.strip('"')
                follow_up_questions_list = follow_up_questions_string.split(', ')       
                follow_up_questions = ', '.join(question.strip() for question in follow_up_questions_list)
                
                answer_content = answer.content
                if "demo" in answer_content:
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": "Demo"
                    })
                elif any(keyword in answer_content for keyword in ["contact us", "reach out", "contacting us"]):
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": "Contact_us"
                    })
                    
                elif answer_content in response_templates:
                    return jsonify(response_templates[answer_content])
                
                else:
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": ""
                    })
        
        elif question_from == "docs":

            with_message_history = ChatAssistant.docs_rag(session_id, 
                                                    question,
                                                    url,
                                                    question_from)
        
            docs = ChatAssistant.docs_retriever.invoke(question)
            sources = [doc.metadata['source'] for doc in docs]

            videos_ = ChatAssistant.transcribe_retriever.invoke(question)
            dara = [doc.metadata['source'] for doc in videos_]
            dara = list(set(dara))

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
            
                else:
                    file_path_without_extension = file_path.replace('.md', '')
                    return file_path_without_extension

            sources_with_link = []

            for doc in docs:
                sources_with_link.append(add_website_url(doc.metadata['source']))

            unique_sources_with_link = list(set(sources))


            if question:
                answer = with_message_history.invoke(
                        {"question": question},
                        config={"configurable": {"session_id": session_id}},
                        )
                history.add_user_message(question)
                history.add_ai_message(answer.content)                

                response_templates = {
                    "Block_msg": {
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "answer": "Schedule a demo with us so we can address your questions. Thank you.",
                        "sources": "",
                        "video_sources":"",
                        "intent": "Block_message"
                    },
                    "Demo": {
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "answer": "Thank you for your interest in scheduling a demo with us. Kindly provide the following details, and our expert will promptly reach out to you. We are eager to demonstrate how our platform can fulfill your requirements.",
                        "sources": "",
                        "video_sources":"",
                        "intent": "Demo"
                    },
                    "Aboutpersoninfo": {
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "answer": "For information about the people working at WaveMaker, please check out our website.",
                        "sources": "",
                        "video_sources":"",
                        "intent": "Aboutpersoninfo"
                    },
                    "Outofwavemaker": {
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "answer": "Sorry,I'm here to provide information about WaveMaker. If you have any questions or need assistance with our platform, feel free to ask. How can I assist you today?",
                        "sources": "",
                        "video_sources":"",
                        "intent": "Outofwavemaker"
                    }
                }
                
                #----
                generator = FollowUpQuestionGenerator()
                follow_up_questions = generator.generate_followup_questions(question)
                follow_up_questions_string = follow_up_questions.strip('"')
                follow_up_questions_list = follow_up_questions_string.split(', ')       
                follow_up_questions = ', '.join(question.strip() for question in follow_up_questions_list)
                
                answer_content = answer.content
                if "demo" in answer_content:
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": "Demo"
                    })
                elif any(keyword in answer_content for keyword in ["contact us", "reach out", "contacting us"]):
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": "Contact_us"
                    })
                    
                elif answer_content in response_templates:
                    return jsonify(response_templates[answer_content])
                
                else:
                    return jsonify({
                        "response_from": "RAG",
                        "faq_id": "",
                        "question": "",
                        "follow_up_questions": follow_up_questions,
                        "answer": answer_content,
                        "sources": unique_sources_with_link,
                        "video_sources":dara,
                        "intent": ""
                    })
        
        elif question_from == "platform":
            response = ChatAssistant.questiongen(question)
            return response

                                                
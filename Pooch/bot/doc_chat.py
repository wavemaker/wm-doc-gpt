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
from langchain import PromptTemplate, LLMChain
from Pooch.helper.prepare_db import PrepareVectorDB
from langchain_together import ChatTogether
from Pooch.helper.followup_question_gen import FollowUpQuestionGenerator
import json
import re
import os
from flask import jsonify
import logging
from Pooch.config.config import( 
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

embeddings = OpenAIEmbeddings()


class WMAssistant:
    _instance = None

    website_loaded_chunks = None
    docs_loaded_chunks = None
    
    website_keyword_retriever = None
    docs_keyword_retriever = None
    
    transcribe_retriever = None
    website_retriever = None
    docs_retriever = None

    def __new__(cls, *args, **kwargs):
        """
        Creates or returns the singleton instance of WMAssistant.

        Returns:
            WMAssistant: The singleton instance of WMAssistant.
        """
        if not cls._instance:
            cls._instance = super(WMAssistant, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initializes the WMAssistant instance by loading chunks and setting up retrievers.
        """
        if self.website_loaded_chunks is None:
            self.load_website_chunks()
        
        if self.docs_loaded_chunks is None:
            self.load_docs_chunks()
        
        if self.website_retriever is None or self.transcribe_retriever is None or self.docs_retriever is None:
            self.website_retriever = self.create_retriever(WEBSITE_COLLECTION)
            logging.info(f"website_retriever: {self.website_retriever}")
            
            self.transcribe_retriever = self.create_retriever(VIDEO_COLLECTION)
            logging.info(f"transcribe_retriever: {self.transcribe_retriever}")
            
            self.docs_retriever = self.create_retriever(DOCS_COLLECTION)
            logging.info(f"docs_retriever: {self.docs_retriever}")

        if self.website_keyword_retriever is None:
            self.website_keyword_retriever = self._initialize_keyword_retriever(self.website_keyword_retriever, self.website_loaded_chunks)
            logging.info(f"website_keyword_retriever: {self.website_keyword_retriever}")

        if self.docs_keyword_retriever is None:
            self.docs_keyword_retriever = self._initialize_keyword_retriever(self.docs_keyword_retriever, self.docs_loaded_chunks)
            logging.info(f"docs_keyword_retriever: {self.website_keyword_retriever}")

    @classmethod
    def load_website_chunks(cls):
        """
        Loads and chunks the website data for retrieval.

        This method initializes the `website_loaded_chunks` attribute by loading and chunking
        documents from the specified website collection.

        Notes:
            This method is called only if `website_loaded_chunks` is `None`.
        """
        if cls.website_loaded_chunks is None:
            read_docs = PrepareVectorDB(WEBSITE_DATA_LOC, WEBSITE_COLLECTION)
            data = read_docs.load_data()
            cls.website_loaded_chunks = read_docs.chunk_documents()

    @classmethod
    def load_docs_chunks(cls):
        """
        Loads and chunks the document data for retrieval.

        This method initializes the `docs_loaded_chunks` attribute by loading and chunking
        documents from the specified docs collection.

        Notes:
            This method is called only if `docs_loaded_chunks` is `None`.
        """
        if cls.docs_loaded_chunks is None:
            docs = PrepareVectorDB(DOCS_DATA_LOC, DOCS_COLLECTION)
            docs_data = docs.load_data()
            cls.docs_loaded_chunks = docs.chunk_documents()

    @staticmethod
    def _get_llm():
        """
        Retrieves the language model based on the environment configuration.

        Returns:
            ChatTogether or ChatOpenAI: The language model instance.

        Notes:
            - Uses `Llama` model if the `MODEL_CHOICE` environment variable is set to 'Llama'.
            - Otherwise, uses the `OpenAI` model.
        """
        model_choice = os.getenv('MODEL_CHOICE')

        if model_choice == 'Llama':
            logging.info("Llama is being used")
            return ChatTogether(
                together_api_key=os.getenv('TOGETHER_API'),
                model="meta-llama/Llama-3-70b-chat-hf",
                max_tokens=500
            )
        else:
            logging.info("OpenAI is being used")
            return ChatOpenAI(
                model_name=MODEL,
                temperature=TEMPERATURE,
                max_tokens=500
            )

    @staticmethod
    def create_retriever(collection_name):
        """
        Creates a retriever instance for the specified collection.

        Args:
            collection_name (str): The name of the collection for which the retriever is created.

        Returns:
            Qdrant: The retriever instance.

        Notes:
            - Uses different search arguments for video collections.
        """
        search_kwargs = {"score_threshold": 0.85, "k": 2} if collection_name == VIDEO_COLLECTION else {}
        
        logging.info(f"Creating retriever for collection: {collection_name}")
        logging.info(f"Search kwargs: {search_kwargs}")
        
        retriever = Qdrant(
            client=CUSTOM_QDRANT_CLIENT,
            embeddings=embeddings,
            collection_name=collection_name
        ).as_retriever(search_kwargs=search_kwargs)
        
        logging.debug(f"Created retriever: {retriever}")
        return retriever

    @staticmethod
    def _initialize_keyword_retriever(keyword_retriever, loaded_chunks):
        """
        Initializes a keyword retriever with the specified chunks.

        Args:
            keyword_retriever: The keyword retriever instance to be initialized.
            loaded_chunks: The chunks of documents to be used for initialization.

        Returns:
            BM25Retriever: The initialized keyword retriever instance.

        Notes:
            - Initializes `keyword_retriever` with BM25Retriever if it is `None`.
            - Sets `k` parameter to 3.
        """
        if keyword_retriever is None:
            keyword_retriever = BM25Retriever.from_documents(loaded_chunks)
            keyword_retriever.k = 3
        return keyword_retriever

    @staticmethod
    def _create_ensemble_retriever(retrievers, weights):
        """
        Creates an ensemble retriever from a list of retrievers and their corresponding weights.

        Args:
            retrievers (list): A list of retriever instances.
            weights (list): A list of weights for each retriever.

        Returns:
            EnsembleRetriever: The ensemble retriever instance.
        """
        return EnsembleRetriever(
            retrievers=retrievers,
            weights=weights,
            return_source_documents=True
        )

    @staticmethod
    def _create_rag_chain(service_mode, llm, ensemble_retriever):
        """
        Creates a RAG (Retrieval-Augmented Generation) chain based on the service mode.

        Args:
            service_mode (str): The service mode, either "website" or "docs".
            llm: The language model instance.
            ensemble_retriever: The ensemble retriever instance.

        Returns:
            Runnable: The RAG chain with message history.
        """
        if service_mode == "website":
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", WEBSITE_SYSTEM_MSG),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ]
            )
        elif service_mode == "docs":
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
        return rag_chain

    @staticmethod
    def get_message_history(session_id, url):
        """
        Retrieves the message history for a specific session.

        Args:
            session_id (str): The session identifier.
            url (str): The URL associated with the session.

        Returns:
            RedisChatMessageHistory: The message history instance.
        """
        return RedisChatMessageHistory(session_id, url=url)

    @staticmethod
    def create_rag(session_id, question, url, service_mode, vectorstore_retriever, keyword_retriever):
        """
        Creates a RAG (Retrieval-Augmented Generation) chain with message history for a specific session.

        Args:
            session_id (str): The session identifier.
            question (str): The question to be answered.
            url (str): The URL associated with the question.
            service_mode (str): The service mode, either "website" or "docs".
            vectorstore_retriever: The vector store retriever instance.
            keyword_retriever: The keyword retriever instance.

        Returns:
            RunnableWithMessageHistory: The RAG chain with message history.
        """
        llm = WMAssistant._get_llm()
        ensemble_retriever = WMAssistant._create_ensemble_retriever(
            retrievers=[vectorstore_retriever, keyword_retriever],
            weights=[0.6, 0.4]
        )
        rag_chain = WMAssistant._create_rag_chain(service_mode, llm, ensemble_retriever)
        def message_history_callable():
            return WMAssistant.get_message_history(session_id, url)
        with_message_history = RunnableWithMessageHistory(
            rag_chain,
            message_history_callable,
            input_messages_key="question",
            history_messages_key="chat_history"
        )
        return with_message_history

    @staticmethod
    def retrieve_results(retriever, question):
        """
        Retrieves results from the specified retriever based on the given question.

        Args:
            retriever: The retriever instance to use.
            question (str): The question to be asked.

        Returns:
            list: A list of source documents related to the question.

        Notes:
            - Logs an error if an exception occurs during retrieval.
        """
        try:
            results = retriever.invoke(question)
            return [doc.metadata['source'] for doc in results]
        except Exception as e:
            logging.error(f"Error retrieving results: {e}")
            return []

    @staticmethod
    def website_pooch(session_id, question, url, service_mode):
        """
        Creates a RAG chain for the website service mode.

        Args:
            session_id (str): The session identifier.
            question (str): The question to be answered.
            url (str): The URL associated with the question.
            service_mode (str): The service mode, "website".

        Returns:
            RunnableWithMessageHistory: The RAG chain with message history for the website.
        """
        vectorstore_retriever = WMAssistant().website_retriever
        keyword_retriever = WMAssistant().website_keyword_retriever
        return WMAssistant.create_rag(session_id, question, url, service_mode, vectorstore_retriever, keyword_retriever)

    @staticmethod
    def docs_pooch(session_id, question, url, service_mode):
        """
        Creates a RAG chain for the docs service mode.

        Args:
            session_id (str): The session identifier.
            question (str): The question to be answered.
            url (str): The URL associated with the question.
            service_mode (str): The service mode, "docs".

        Returns:
            RunnableWithMessageHistory: The RAG chain with message history for the docs.
        """
        docsvectorstoreretriever = WMAssistant().docs_retriever
        docskeywordretriever = WMAssistant().docs_keyword_retriever
        return WMAssistant.create_rag(session_id, question, url, service_mode, docsvectorstoreretriever, docskeywordretriever)


    @staticmethod
    def generate_response(answer_content, question, sources, video_sources):
        """
        Generates a response object based on the provided answer content, question, sources, and video sources.

        Args:
            answer_content (str): The content of the answer to be included in the response.
            question (str): The question that prompted the answer.
            sources (list): A list of sources related to the answer.
            video_sources (list): A list of video sources related to the answer.

        Returns:
            dict: A response dictionary containing the response details. The dictionary includes the following keys:
                - response_from (str): Indicates the source of the response (e.g., "RAG").
                - faq_id (str): An identifier for frequently asked questions.
                - question (str): The question that prompted the response.
                - answer (str): The answer content.
                - follow_up_questions (str): A string of follow-up questions generated based on the question.
                - sources (list): A list of sources related to the answer.
                - video_sources (list): A list of video sources related to the answer.
                - intent (str): The intent of the response (e.g., "Demo", "Contact_us", "Block_message").

        Notes:
            - The method uses a predefined set of response templates for specific answers.
            - It attempts to generate follow-up questions using the `FollowUpQuestionGenerator`. If an error occurs during this process, it logs the error and sets `follow_up_questions` to `None`.
            - If the `answer_content` contains the keyword "demo", the method returns a response with the intent "Demo".
            - If the `answer_content` contains keywords related to contacting, it returns a response with the intent "Contact_us".
            - If the `answer_content` matches any predefined response templates, it returns the corresponding template.
            - For other answer content, it returns a default response with the provided answer content, follow-up questions, sources, and video sources.
        """

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
                "answer": "Sorry, I'm here to provide information about WaveMaker. If you have any questions or need assistance with our platform, feel free to ask. How can I assist you today?",
                "sources": "",
                "intent": "Outofwavemaker"
            }
        }

        try:
            generator = FollowUpQuestionGenerator()
            follow_up_questions = generator.generate_followup_questions(question)
            follow_up_questions = ', '.join(q.strip() for q in follow_up_questions.strip('"').split(', '))

        except Exception as e:
            logging.error(f"Error generating follow-up questions: {e}")
            follow_up_questions = None

        if "demo" in answer_content:
            return {
                "response_from": "RAG",
                "faq_id": "",
                "question": "",
                "follow_up_questions": follow_up_questions,
                "answer": answer_content,
                "sources": sources,
                "video_sources": video_sources,
                "intent": "Demo"
            }
        elif any(keyword in answer_content for keyword in ["contact us", "reach out", "contacting us"]):
            return {
                "response_from": "RAG",
                "faq_id": "",
                "question": "",
                "follow_up_questions": follow_up_questions,
                "answer": answer_content,
                "sources": sources,
                "video_sources": video_sources,
                "intent": "Contact_us"
            }
        elif answer_content in response_templates:
            return response_templates[answer_content]
        else:
            return {
                "response_from": "RAG",
                "faq_id": "",
                "question": "",
                "follow_up_questions": follow_up_questions,
                "answer": answer_content,
                "sources": sources,
                "video_sources": video_sources,
                "intent": ""
            }


    @staticmethod
    def add_website_url(file_path):
        """
        Converts a local file path to a full website URL based on its structure.

        Args:
            file_path (str): The local file path to be converted into a website URL.

        Returns:
            str: The corresponding website URL for the given file path. 

        Notes:
            - If the file path contains 'learn' and 'docs', it is assumed to be a documentation page under the 'learn' section.
              The URL will be constructed to point to the 'learn' section of the website.
            - If the file path contains 'blog' and 'docs', it is assumed to be a blog post. The URL will be constructed to include
              the blog section, formatted date, and title of the blog post.
            - For other file paths, the method simply removes the '.md' extension and returns the modified path.
        """
        docs_url = 'https://docs.wavemaker.com'

        if 'learn' in file_path and 'docs' in file_path:
            trimmed_path = file_path.replace('.md', '')
            return f"{docs_url}/learn{trimmed_path.split('learn')[1]}"
        elif 'blog' in file_path and 'docs' in file_path:
            date_parts = file_path.split('/')[-1].split('-')
            formatted_date = '/'.join(date_parts[:3]) if len(date_parts) >= 3 else ""
            title_parts = file_path.split('/')[-1].split('-')
            title = '-'.join(title_parts[3:]) if len(title_parts) >= 4 else ""
            title = title[:-3] if title.endswith('.md') else title
            return f"{docs_url}/learn/blog/{formatted_date}/{title}/"
        else:
            return file_path.replace('.md', '')

    @staticmethod
    def convert_links_and_add_timestamp(documents):
        converted_links = []

        for doc in documents:
            source_url = doc.metadata['source']
            new_base_url = source_url.replace("https://embed.app.guidde.com", "https://app.guidde.com/share")

            page_content = doc.page_content
            timestamp_match = re.search(r'\d{2}:\d{2}', page_content)
            
            if timestamp_match:
                timestamp_str = timestamp_match.group()
                minutes, seconds = map(int, timestamp_str.split(':'))
                total_seconds = minutes * 60 + seconds
            else:
                total_seconds = 0

            new_url = f"{new_base_url}?origin=ccq9JWcblzMXt0obBzCh79ljH20p2&t={total_seconds}"

            converted_links.append(new_url)

        return converted_links

    @staticmethod
    def process_question(session_id, question, url, question_from):
        """
        Processes a question based on the specified source and retrieves relevant information.

        Args:
            session_id (str): The unique identifier for the current user session.
            question (str): The question to be processed.
            url (str): The URL related to the question, if applicable.
            question_from (str): The source of the question, either "website" or "docs".

        Returns:
            tuple: A tuple containing:
                - with_message_history: An instance of RunnableWithMessageHistory, used to keep track of the conversation history.
                - retriever: The retriever instance used to fetch relevant documents based on the question.
                - sources (list): A list of URLs or source identifiers related to the retrieved documents.
                - video_sources (list): A list of URLs or source identifiers related to the retrieved videos.

        Raises:
            ValueError: If the `question_from` parameter is not "website" or "docs".
        """
        if question_from == "website":
            with_message_history = WMAssistant.website_pooch(session_id, 
                                                             question, 
                                                             url, 
                                                             question_from)
            retriever = WMAssistant().website_retriever
        
        elif question_from == "docs":
            with_message_history = WMAssistant.docs_pooch(session_id, 
                                                          question, 
                                                          url, 
                                                          question_from)
            retriever = WMAssistant().docs_retriever

        else:
            raise ValueError(f"Invalid value for `question_from`: {question_from}")

        # Retrieve documents related to the question using the chosen retriever
        docs = retriever.invoke(question)
        sources = [WMAssistant.add_website_url(doc.metadata['source']) for doc in docs]

        # Retrieve videos related to the question using the transcribe retriever
        videos_ = WMAssistant().transcribe_retriever.invoke(question)
        video_sources = list(set(WMAssistant.convert_links_and_add_timestamp(videos_)))

        return with_message_history, retriever, sources, video_sources


    @staticmethod
    def answer_question(session_id, question, url, question_from):
        """
        Processes and answers a given question based on its source (website or docs) and the provided context.

        This method checks the validity of the question source, retrieves relevant information, generates an answer,
        and records the interaction in a chat history. It handles errors gracefully and returns a JSON response.

        Args:
            session_id (str, optional): The ID of the user session for tracking conversation history.
            question (str, optional): The question to be answered.
            url (str): The URL related to the question, if applicable.
            question_from (str): The source of the question, which can be either "website" or "docs".

        Returns:
            Response: A JSON response containing the generated answer and relevant metadata, or an error message.

        Raises:
            ValueError: If the `question_from` argument is neither "website" nor "docs".
        """
        if question_from not in ["website", "docs"]:
            return jsonify({"error": "Invalid question source."})

        with_message_history, retriever, sources, video_sources = WMAssistant.process_question(
            session_id, question, url, question_from
        )

        if question:
            try:
                answer = with_message_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                )
                
                history = RedisChatMessageHistory(session_id, url=url, ttl=60)
                history.add_user_message(question)
                history.add_ai_message(answer.content)

                response = WMAssistant.generate_response(answer.content, question, sources, video_sources)
                return jsonify(response)

            except Exception as e:
                logging.error(f"Error handling question: {e}")
                return jsonify({"error": "An error occurred while processing the question."})
    
    @staticmethod
    def _generate_questions(llm, keyword, chunks):
        """
        Generates questions based on the retrieved chunks using a language model.

        Args:
            llm: The language model instance.
            keyword (str): The keyword to generate questions for.
            chunks: The retrieved document chunks.

        Returns:
            str: Generated questions as a response from the language model.
        """
        # Define the prompt template for generating questions
        prompt = PromptTemplate(
            template=QUESTION_GEN_SYSTEM_MSG,
            input_variables=['keyword', 'chunks']
        )

        llm_chain = LLMChain(
            prompt=prompt,
            llm=llm
        )

        return llm_chain.run(keyword=keyword, chunks=chunks)

    @staticmethod
    def questiongen(keyword):
        """
        Generates follow-up questions based on the given keyword by retrieving relevant chunks
        and passing them through a language model.

        Args:
            keyword (str): The keyword to generate questions for.

        Returns:
            str: Generated questions as a response from the language model.
        """
        llm = WMAssistant._get_llm()

        docs_retriever = WMAssistant().docs_retriever
        chunks = docs_retriever.invoke(keyword)
        response = WMAssistant._generate_questions(llm, keyword, chunks)
        return response                                             
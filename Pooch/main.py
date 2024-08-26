import os
import csv
import uuid
import json
from datetime import timedelta
import logging
from flask import Flask, request, jsonify, session
from langchain_community.chat_message_histories import RedisChatMessageHistory
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from qdrant_client import models, QdrantClient
from Pooch.helper.scrapper import Scraper
from Pooch.helper.prepare_db import PrepareVectorDB
from Pooch.bot.faq_chat import CollectionUploadChecker
from Pooch.bot.faq_chat import search
from Pooch.bot.doc_chat import WMAssistant
from Pooch.helper.prepare_db import DeleteDuplicates
from Pooch.helper.prepare_db import PrepareAndSaveScrappedData
from Pooch.helper.prepare_db import SemanticSearch
from Pooch.helper.scrapper import ScrapePDFAndSave
from Pooch.helper.semantic_router import query_route
from Pooch.helper.followup_question_gen import FollowUpQuestionGenerator
from Pooch.helper.doc_summary import DocSummary
from Pooch.helper.preprocess_video import process_pdf_directory, ingest_markdown_content
from Pooch.config.config import (
        #COLLECTION_NAME, 
        # DATA_LOC,
        REDIS_URL,
        FAQ_LOC,
        FAQ_COLLECTION_NAME,
        GITHUB_DOCS,
        WAVEMAKER_WEBSITE,
        WAVEMAKER_AI,
        FILES_FROM_REQUEST,
        UPLOAD_SCRAPPED_DATA,
        CUSTOM_QDRANT_CLIENT,
        VIDEO_COLLECTION,
        DOCS_COLLECTION,
        WAVEMAKER_WEBSITE,
        WEBSITE_COLLECTION,
        VIDEO_SOURCES
)

app = Flask(__name__)
# app.config.from_object(files_)

@app.route('/answer', methods=['POST'])
def answer_question():
    headers = dict(request.headers)
    logging.info(f"Request headers: {headers}")

    user_id = request.headers.get('Uuid')
    logging.info(f"User ID: {user_id}")

    data = request.json
    question = data.get('question')
    service_mode = request.args.get('service_mode')

    if question.endswith('?'):
        question = question.rstrip('?')

    intent = query_route(question)

    response_template = {
        "response_from": "ROUTER",
        "faq_id": "",
        "question": "",
        "follow_up_questions": "",
        "answer": "",
        "sources": "",
        "intent": ""
    }

    try:
        if intent == "Greeting":
            response_template["answer"] = "Hello, I'm Pooch! How can I assist you today? If you have any questions or need assistance related to WaveMaker, feel free to ask. I'm here to provide information and support regarding WaveMaker and its features."
            response_template["intent"] = intent
            generator = FollowUpQuestionGenerator()
            follow_up_questions = generator.generate_followup_questions(question)
            follow_up_questions_string = follow_up_questions.strip('"')
            follow_up_questions_list = follow_up_questions_string.split(', ')        
            follow_up_questions_str = ', '.join(question.strip() for question in follow_up_questions_list)
            response_template["follow_up_questions"] = follow_up_questions_str
            return jsonify(response_template)
        
        elif intent == "Demo":
            response_template["answer"] = "Thank you for your interest in scheduling a demo with us. Kindly provide the following details, and our expert will promptly reach out to you. We are eager to demonstrate how our platform can fulfill your requirements."
            response_template["intent"] = intent
            return jsonify(response_template)
        
        elif intent == "Contact_us":
            response_template["answer"] = "Thank you for reaching out to us. Please provide the following details, and our team will get back to you shortly. We're here to assist you and answer any questions you may have."
            response_template["intent"] = intent
            return jsonify(response_template)

        elif intent == "Name":
            response_template["answer"] = "Well, it's cute for one. I am trained to fetch stuff well, for another. And there's also an Indian wordplay where pooch means 'ask' in Hindi."
            response_template["intent"] = intent
            generator = FollowUpQuestionGenerator()
            follow_up_questions = generator.generate_followup_questions(question)
            follow_up_questions_string = follow_up_questions.strip('"')
            follow_up_questions_list = follow_up_questions_string.split(', ')       
            follow_up_questions_str = ', '.join(question.strip() for question in follow_up_questions_list)
            response_template["follow_up_questions"] = follow_up_questions_str
            return jsonify(response_template)
        
        elif intent == "Ragpipe":
            pass

        logging.info("Query searching in the FAQ collection is started")
        sim_results = search(question)
        history = RedisChatMessageHistory(user_id, url=REDIS_URL)

        for hit in sim_results:
            if hit.score < 0.85:
                logging.info("Rag flow initialized for the query")
                assistant = WMAssistant()
                answer = assistant.answer_question(user_id, question, REDIS_URL, service_mode)
                return answer

            else:
                curated_ans = sim_results[0].payload['answer']
                generator = FollowUpQuestionGenerator()
                follow_up_questions = generator.generate_followup_questions(question)
                history.add_user_message(question)
                history.add_ai_message(curated_ans)
                return jsonify({
                    "response_from": "FAQ",
                    "faq_id": sim_results[0].id,
                    'question': sim_results[0].payload['question'], 
                    "follow_up_questions": follow_up_questions,
                    'answer': curated_ans,
                    "sources": "",
                    "intent": ""
                })

    except Exception as e:
        logging.error(f"Error processing the request: {e}")
        return jsonify({"error": "An error occurred while processing the request."}), 500


@app.route('/ingest', methods=['POST', 'PUT', 'DELETE'])
def handle_ingestion():
    # group = request.form['group']
    group = request.args.get('group')


    if group is None:
        return jsonify({"error": "Missing required query parameters"}), 400
    
    if group == "FAQ":
        try:
            if request.method == "POST":
                file = request.files.get('file')
                if file :
                    filename = secure_filename(file.filename)
                    faq_data = "temp_faq_datastore"
                    if not os.path.exists(faq_data):
                        os.makedirs(faq_data)
                    
                    json_filename = os.path.join(faq_data, filename)
                    file.save(json_filename)
                else:    
                    json_data = request.json
                    faq_data = "temp_faq_datastore"
                
                    if not os.path.exists(faq_data):
                        os.makedirs(faq_data)
                    
                    json_filename = os.path.join(faq_data, "data.json")
                    with open(json_filename, 'w') as json_file:
                        json.dump([json_data], json_file, indent=4)

                faq_collection = CollectionUploadChecker(FAQ_COLLECTION_NAME,
                                                         json_filename)
                vectors = faq_collection.check_collection_upload_data()

                os.remove(json_filename)
                
                if vectors :
                    response_data = {"message": f"Data ingested successfully with collection: {FAQ_COLLECTION_NAME}"}
                    return jsonify(response_data)
                else:
                    response_data = {"message": f"Data ingestion failed with collection: {FAQ_COLLECTION_NAME}"}
                    return jsonify(response_data)

            elif request.method == 'PUT':

                json_data = request.json
                faq_data = "temp_faq_datastore"
                
                if not os.path.exists(faq_data):
                    os.makedirs(faq_data)
                
                json_filename = os.path.join(faq_data, "data.json")
                with open(json_filename, 'w') as json_file:
                    json.dump([json_data], json_file, indent=4)

                faq_collection = CollectionUploadChecker(FAQ_COLLECTION_NAME,
                                                         json_filename)
                vectors = faq_collection.check_collection_upload_data()

                os.remove(json_filename)
                
                if vectors :
                    response_data = {"message": f"Data updated successfully with collection: {FAQ_COLLECTION_NAME}"}
                    return jsonify(response_data)
                else:
                    response_data = {"message": f"Data update failed with collection: {FAQ_COLLECTION_NAME}"}
                    return jsonify(response_data)
                
            elif request.method == 'DELETE':

                id = request.args.get("id")

                delete_operation = SemanticSearch()
                delete_data = delete_operation.delete_qan(id)
                
                if delete_data:
                    response_data = {"message" : "Data deleted successfully"}
                    return jsonify(response_data)
                
                else:
                    response_data = {"message" : "Data deletion failed"}
                    return jsonify(response_data)
        
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    elif group == "docs":
        try:
            read_docs = PrepareVectorDB(GITHUB_DOCS, DOCS_COLLECTION)
            stored_vector = read_docs.prepare_and_save_vectordb()
            
            if stored_vector is not None:
                response_data = {"message": f"Docs data ingested successfully with collection: {DOCS_COLLECTION}"}
            else:
                response_data = {"message": f"Docs data ingestion failed with collection: {DOCS_COLLECTION}"}
            
            return jsonify(response_data)
        
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    
    elif group == "website":
        try:
            for docs_source in [GITHUB_DOCS, WAVEMAKER_WEBSITE, WAVEMAKER_AI]:
                read_docs = PrepareVectorDB(docs_source, WEBSITE_COLLECTION)
                stored_vector = read_docs.prepare_and_save_vectordb()

                if stored_vector is None:
                    response_data = {"message": f"Docs data ingestion failed with collection: {WEBSITE_COLLECTION}"}
                    return jsonify(response_data)

            response_data = {"message": f"Website data ingested successfully with collection: {WEBSITE_COLLECTION}"}
            return jsonify(response_data)

        except Exception as e:
            response_data = {"message": f"An error occurred: {str(e)}"}
            return jsonify(response_data)
    
    elif group == "video_data":
        try:
            results = process_pdf_directory(VIDEO_SOURCES)
            response, status_code = ingest_markdown_content(results, VIDEO_COLLECTION)
            return response, status_code

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/scrape', methods=['POST', 'PUT', 'DELETE'])
def scrape():
    logging.info("scraping is started!")
    urls = []
    
    if 'url' in request.form:
        urls.append(request.form['url'])

    elif 'file' in request.files:
        file = request.files['file']
        
        #======Store the files from the request============#
        if not os.path.exists(FILES_FROM_REQUEST):
            os.makedirs(FILES_FROM_REQUEST)

        file.save(os.path.join(FILES_FROM_REQUEST, file.filename))
        with open(os.path.join(FILES_FROM_REQUEST, file.filename), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            urls.extend(row['URL'] for row in reader)
    else:
        return jsonify({"error": "No URL or CSV file provided"}), 400

    #======Store the scraped files in the s3============#
    folder_path = os.path.join(os.getcwd(), UPLOAD_SCRAPPED_DATA) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    if request.method == "POST":
        try:
            success_flag = False
            
            for url in urls:
                logging.info("inside the pdf")
                if url.endswith(".pdf"):
                    try:
                        logging.info("url",url)
                        converter = ScrapePDFAndSave(url)
                        success, pdf_text, pdf_filename = converter.read_pdf_from_web()

                        parsed_url = urlparse(url)
                        filename = parsed_url.path.strip('/').replace('/', '-') + ".md"
                        logging.info("filename",filename)
                        file_path = os.path.join(folder_path, filename)

                        with open(file_path, 'w') as file:
                            file.write(pdf_text)
                        
                        logging.info("url",url)
                        read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA, WEBSITE_COLLECTION)
                        stored_vector = read_docs.prepare_and_save_scrapped_data(url)

                        if stored_vector:
                            wavemaker_file_path = os.path.join(WAVEMAKER_WEBSITE, filename)
                            scrapped_data_path = os.path.join(UPLOAD_SCRAPPED_DATA, filename)

                            if os.path.exists(wavemaker_file_path):
                                os.remove(wavemaker_file_path)
                            
                            with open(wavemaker_file_path, 'w') as file:
                                file.write(pdf_text)
                            
                            success_flag = True
                        

                        if os.path.exists(scrapped_data_path):
                            os.remove(scrapped_data_path)

                        # Set the flag to True if the process was successful for at least one URL
                        success_flag = True
                        
                    except Exception as e:
                        logging.error(f"PDF conversion failed {e}")
                        return("An error occurred:", e)

                else :
                    parsed_html, error = Scraper.scrape_website(url)
                
                    if error:
                        return jsonify({"error": str(error)}), 500
                    
                    parsed_url = urlparse(url)
                    filename = parsed_url.path.strip('/').replace('/', '-') + ".md"
                    file_path = os.path.join(folder_path, filename)

                    with open(file_path, 'w') as file:
                        file.write(parsed_html.cleaned_text)
                    
                
                    read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA, WEBSITE_COLLECTION)
                    stored_vector = read_docs.prepare_and_save_scrapped_data(url)

                    if stored_vector:
                        wavemaker_file_path = os.path.join(WAVEMAKER_WEBSITE, filename)
                        scrapped_data_path = os.path.join(UPLOAD_SCRAPPED_DATA, filename)
                        
                        if os.path.exists(wavemaker_file_path):
                            os.remove(wavemaker_file_path)
                        with open(wavemaker_file_path, 'w') as file:
                            file.write(parsed_html.cleaned_text)
                        

                        if os.path.exists(scrapped_data_path):
                            os.remove(scrapped_data_path)

                        # Set the flag to True if the process was successful for at least one URL
                        success_flag = True

            if success_flag:
                response_data = {"message": f"Website data ingested successfully with collection: {WEBSITE_COLLECTION}"}
                return jsonify(response_data), 200
            else:
                response_data = {"message": f"Website data ingestion failed with collection: {WEBSITE_COLLECTION}"}
                return jsonify(response_data), 500

        except Exception as e:
            logging.error(f"Error ingesting website data: {e}")
            response_data = {"message": f"Internal server error: {e}"}
            return jsonify(response_data), 500
    
    elif request.method == "PUT":
        try:
            success_flag = False
            
            for url in urls:
                parsed_html, error = Scraper.scrape_website(url)
                
                if error:
                    logging.error(f"Error scraping website data: {error}")
                    continue

                parsed_url = urlparse(url)
                filename = parsed_url.path.strip('/').replace('/', '-') + ".md"
                file_path = os.path.join(folder_path, filename)

                if os.path.exists(file_path):
                    os.remove(file_path)
                with open(file_path, 'w') as file:
                    file.write(parsed_html.cleaned_text)
                
                wavemaker_file_path = os.path.join(WAVEMAKER_WEBSITE, filename)
                
                if os.path.exists(wavemaker_file_path):
                    os.remove(wavemaker_file_path)
                with open(wavemaker_file_path, 'w') as file:
                    file.write(parsed_html.cleaned_text)
                
                full_url = WAVEMAKER_WEBSITE + '/' + filename

                delete_duplicates = DeleteDuplicates(url, WEBSITE_COLLECTION)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(collection_name=WEBSITE_COLLECTION)
                result_id = delete_duplicates.get_id_from_source(all_data, delete_duplicates.url)

                if result_id is None:
                    try:
                        read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA)
                        stored_vector = read_docs.prepare_and_save_scrapped_data(url)

                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                        if stored_vector is not None:
                            success_flag = True
                        else:
                            logging.error(f"Website data ingestion failed with collection: {WEBSITE_COLLECTION}")
                        
                    except Exception as e:
                        logging.error(f"Error ingesting website data: {e}")
                    
                else:
                    try:
                        delete_duplicates.delete_vector(result_id)
                        read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA)
                        stored_vector = read_docs.prepare_and_save_scrapped_data(url)

                        if os.path.exists(file_path):
                            os.remove(file_path)

                        if stored_vector is not None:
                            success_flag = True
                        else:
                            logging.error(f"Website data ingestion failed with collection: {WEBSITE_COLLECTION}")
                        
                    except Exception as e:
                        logging.error(f"Error ingesting website data: {e}")
            
            if success_flag:
                response_data = {"message": f"Website data ingested and updated successfully with collection: {WEBSITE_COLLECTION}"}
                return jsonify(response_data), 200
            else:
                response_data = {"message": f"Website data ingestion failed with collection: {WEBSITE_COLLECTION}"}
                return jsonify(response_data), 500

        except Exception as e:
            logging.error(f"Internal server error: {e}")
            response_data = {"message": f"Internal server error: {e}"}
            return jsonify(response_data), 500

            
    elif request.method == "DELETE":
        try:
            success_flag = False
            
            for url in urls:
                parsed_url = urlparse(url)
                filename = parsed_url.path.strip('/').replace('/', '-') + ".md"
                file_path = os.path.join(folder_path, filename)

                if os.path.exists(file_path):
                    os.remove(file_path)
                
                wavemaker_file_path = os.path.join(UPLOAD_SCRAPPED_DATA, filename)
                if os.path.exists(wavemaker_file_path):
                    os.remove(wavemaker_file_path)
                
                full_url = WAVEMAKER_WEBSITE + '/' + filename

                delete_duplicates = DeleteDuplicates(url, WEBSITE_COLLECTION)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(WEBSITE_COLLECTION)
                result_id = delete_duplicates.get_id_from_source(all_data, delete_duplicates.url)

                if result_id is None:
                   return jsonify({"Message": "No related content is present in the DB!"})

                deleted_data = delete_duplicates.delete_vector(result_id)

                if os.path.exists(full_url):
                    os.remove(full_url)
                
                if deleted_data :
                    success_flag = True
                    logging.info("Website data deleted successfully")

            if success_flag:
                response_data = {"message": "Website data deleted successfully"}
                return jsonify(response_data), 200
            else:
                response_data = {"message": "Website data deletion unsuccessful"}
                logging.error("Error deleting website data")
                return jsonify(response_data), 500

        except Exception as e:
            logging.error(f"Error deleting website data: {e}")
            response_data = {"message": f"Internal server error: {e}"}
            return jsonify(response_data), 500
    return jsonify({"message": "Scraping is successfull"}), 200

@app.route('/delete_collection', methods=['GET'])
def del_collection():
    try:
        collection_name = request.args.get('collection')
        if not collection_name:
            return jsonify({"error": "Collection name is required"}), 400

        deleted_col = CUSTOM_QDRANT_CLIENT.delete_collection(collection_name=collection_name)
        
        if deleted_col is None:
            collection_response = {"message": "Collection is not deleted"}
            return jsonify(collection_response)

        collection_response = {"message": "Collection deleted"}
        return jsonify(collection_response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while deleting the collection"}), 500

@app.route('/health', methods=['GET'])
def health():
    health_response = {"message": "Health check successful"}
    return jsonify(health_response)

from Pooch.bot.doc_chat import WMAssistant
@app.route('/questionGen', methods=['GET'])
def queGen():
    data = request.json
    question = data.get('question')
    gen = WMAssistant.questiongen(question)
    return jsonify(gen)

@app.route('/summary', methods=['POST'])
def summary_gen():
        
        logging.info(f"session------> {session}")
        user_id = request.headers.get('Uuid')

        history = RedisChatMessageHistory(user_id, 
                        url=REDIS_URL)

        data = request.json
        url = data.get('url')

        parsed_html, error = Scraper.scrape_website(url)
        generator = DocSummary()
        summary = generator.generate_summary(parsed_html.cleaned_text)
        
        history.add_user_message(url)
        history.add_ai_message(summary)
        return jsonify(summary)

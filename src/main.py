import os
import csv
import uuid
import json
from datetime import timedelta
import logging
from flask import Flask, request, jsonify, session, abort
from langchain_community.chat_message_histories import RedisChatMessageHistory
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from qdrant_client import models, QdrantClient
from src.helper.scrapper import Scraper
from src.helper.prepare_db import PrepareVectorDB
from src.bot.faq_chat import CollectionUploadChecker
from src.bot.faq_chat import search
from src.bot.doc_chat import ChatAssistant
from src.helper.prepare_db import DeleteDuplicates
from src.helper.prepare_db import PrepareAndSaveScrappedData
from src.helper.prepare_db import SemanticSearch
from src.helper.scrapper import ScrapePDFAndSave
from src.helper.semantic_router import query_route
from src.helper.followup_question_gen import FollowUpQuestionGenerator
from src.config.config import (
        COLLECTION_NAME, 
        # DATA_LOC,
        REDIS_URL,
        FAQ_LOC,
        FAQ_COLLECTION_NAME,
        GITHUB_DOCS,
        WAVEMAKER_WEBSITE,
        WAVEMAKER_AI,
        FILES_FROM_REQUEST,
        UPLOAD_SCRAPPED_DATA,
        CUSTOM_QDRANT_CLIENT
)

app = Flask(__name__)

# app.secret_key = os.environ["SECRET_KEY"]

# app.permanent_session_lifetime = timedelta(minutes=5)

@app.route('/answer', methods=['POST'])
def answer_question():
    headers = dict(request.headers)
    logging.info("request_headers",headers)

    user_id = request.headers.get('Uuid')
    
    logging.info("user_id",user_id)

    data = request.json
    question = data.get('question')
    if question.endswith('?'):
        question = question.rstrip('?')

    intent = query_route(question)

    response_template = {
    "response_from": "ROUTER",
    "faq_id": "",
    "question": "",
    "follow_up_questions": "",
    "answer":"",
    "sources": "",
    "intent": ""
    }

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
    history = RedisChatMessageHistory(user_id, 
                                      url=REDIS_URL)

    for hit in sim_results:
        if hit.score < 0.85:
            logging.info("Rag flow initilised for the query")
            assistant = ChatAssistant()
            answer = assistant.answer_question(user_id, question,REDIS_URL)
            return answer
        
        else:
            curatedAns = sim_results[0].payload['answer']
            generator = FollowUpQuestionGenerator()
            follow_up_questions = generator.generate_followup_questions(question)
            history.add_user_message(question)
            history.add_ai_message(curatedAns)
            return jsonify({
                            "response_from":"FAQ",
                            "faq_id":sim_results[0].id,
                            'question':sim_results[0].payload['question'], 
                            "follow_up_questions": follow_up_questions,
                            'answer': sim_results[0].payload['answer'],
                            "sources": "",
                            "intent": ""})

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

    elif group == "docs" or "website" or "ai_website":
        try:
            if group == "docs":
                read_docs = PrepareVectorDB(GITHUB_DOCS)
                stored_vector = read_docs.prepare_and_save_vectordb()
                
                if stored_vector != None:
                    response_data = {"message": f"Docs data ingested successfull with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)
                
                else:
                    response_data = {"message": f"Docs data ingested failed with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)
            
            elif group == "website":
                read_docs = PrepareVectorDB(WAVEMAKER_WEBSITE)
                stored_vector = read_docs.prepare_and_save_vectordb()
                
                if stored_vector != None:
                    response_data = {"message": f"Website data ingested successfull with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)
                
                else:
                    response_data = {"message": f"Website data ingested failed with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)
            
            elif group == "ai_website": 
                read_docs = PrepareVectorDB(WAVEMAKER_AI)
                stored_vector = read_docs.prepare_and_save_vectordb()
                
                if stored_vector != None:
                    response_data = {"message": f"Wavemakerai website data ingested successfull with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)
                
                else:
                    response_data = {"message": f"Wavemakerai website data ingested failed with collection: {COLLECTION_NAME}"}
                    return jsonify(response_data)  
            
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

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
                        read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA)
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
                    
                
                    read_docs = PrepareAndSaveScrappedData(UPLOAD_SCRAPPED_DATA)
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
                response_data = {"message": f"Website data ingested successfully with collection: {COLLECTION_NAME}"}
                return jsonify(response_data), 200
            else:
                response_data = {"message": f"Website data ingestion failed with collection: {COLLECTION_NAME}"}
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

                delete_duplicates = DeleteDuplicates(url, COLLECTION_NAME)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(collection_name=COLLECTION_NAME)
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
                            logging.error(f"Website data ingestion failed with collection: {COLLECTION_NAME}")
                        
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
                            logging.error(f"Website data ingestion failed with collection: {COLLECTION_NAME}")
                        
                    except Exception as e:
                        logging.error(f"Error ingesting website data: {e}")
            
            if success_flag:
                response_data = {"message": f"Website data ingested and updated successfully with collection: {COLLECTION_NAME}"}
                return jsonify(response_data), 200
            else:
                response_data = {"message": f"Website data ingestion failed with collection: {COLLECTION_NAME}"}
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

                delete_duplicates = DeleteDuplicates(url, COLLECTION_NAME)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(COLLECTION_NAME)
                print("all_data:",all_data)
                result_id = delete_duplicates.get_id_from_source(all_data, delete_duplicates.url)
                print("result_id:",result_id)
                deleted_data = delete_duplicates.delete_vector(result_id)
                print("deleted_data:",deleted_data)

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

API_KEYS = os.getenv('API_KEYS') 

def require_api_key(f):
    def decorated_function(*args, **kwargs):
        if 'API-Key' not in request.headers or request.headers['API-Key'] not in API_KEYS:
            abort(401)  # Unauthorized
        return f(*args, **kwargs)
    return decorated_function

@app.route('/insertion', methods=['POST'])
@require_api_key
def insertion():
    if request.is_json:
        data = request.json
        
        if 'Added_files' in data :
            added_files = data['Added_files']

            for source in added_files:
                dele = DeleteDuplicates(source, COLLECTION_NAME)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(collection_name=COLLECTION_NAME,limit = 20)
                result_id = dele.get_id_from_source(all_data, source)

                if result_id:
                    dele.delete_vector(result_id)
                    read_docs = PrepareVectorDB(source)                    
                    stored_vector = read_docs.prepare_and_save_vectordb()

                    # dele.add_to_directory(source)
                    
                    if stored_vector != None:
                        response_data = {"message": f"Docs from the Github added successfully with collection: {COLLECTION_NAME}"}
                        # return jsonify(response_data)
                    
                    else:
                        response_data = {"message": f"Docs from the Github failed with collection: {COLLECTION_NAME}"}
                        # return jsonify(response_data)
                else:
                    read_docs = PrepareVectorDB(source)
                    stored_vector = read_docs.prepare_and_save_vectordb()

                    # dele.add_to_directory(source)
                    
                    if stored_vector != None:
                        response_data = {"message": f"Docs from the Github added successfully with collection: {COLLECTION_NAME}"}
                        # return jsonify(response_data)
                    
                    else:
                        response_data = {"message": f"Docs from the Github failed with collection: {COLLECTION_NAME}"}
                        # return jsonify(response_data)

            response_data = {"message": f"Docs from the Github added successfully with collection: {COLLECTION_NAME}"}
            return jsonify(response_data)

        # elif 'Updated_files' in data:
        #     added_files = data['Updated_files']
        elif 'Deleted_files' in data:

            added_files = data['Deleted_files']

            for source in added_files:
                dele = DeleteDuplicates(source, COLLECTION_NAME)
                all_data, _ = CUSTOM_QDRANT_CLIENT.scroll(collection_name=COLLECTION_NAME,limit = 20)
                result_id = dele.get_id_from_source(all_data, source)

                if result_id:
                    dele.delete_vector(result_id)
            response_data = {"message": "Data deleted successfully"}
            return jsonify(response_data)

@app.route('/delete_collection', methods=['GET'])
def del_collection():
    collection_name = request.args.get('collection')
    CUSTOM_QDRANT_CLIENT.delete_collection(collection_name=collection_name)
    collection_response = {"message": "Collection deleted"}
    return jsonify(collection_response)

@app.route('/health', methods=['GET'])
def health():
    health_response = {"message": "Health check successful"}
    return jsonify(health_response)


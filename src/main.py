from src.helper.prepare_db import PrepareVectorDB
from src.bot.faq_chat import search
from src.bot.doc_chat import ChatAssistant
from flask import Flask, request, jsonify, session
from src.bot.faq_chat import CollectionUploadChecker
from langchain_community.chat_message_histories import RedisChatMessageHistory
import uuid
from datetime import timedelta
import os
import csv
from urllib.parse import urlparse
from src.helper.scrapper import Scraper
import logging
from dotenv import load_dotenv
from src.config.config import (
        COLLECTION_NAME, 
        DATA_LOC,
        REDIS_URL,
        FAQ_LOC,
        FAQ_COLLECTION_NAME
)

app = Flask(__name__)

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(hours=1)

@app.route('/answer', methods=['POST'])
def answer_question():
    if 'user_id' not in session:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
    else:
        user_id = session['user_id']
    data = request.json
    question = data.get('question')

    logging.info("Query searching in the FAQ collection is started")
    sim_results = search(question)
    history = RedisChatMessageHistory(user_id, 
                                      url=REDIS_URL)

    for hit in sim_results:
        if hit.score < 0.80:
            logging.info("Rag flow initilised for the query")
            assistant = ChatAssistant()
            answer = assistant.answer_question(user_id, question,REDIS_URL)
            docs = assistant.get_sources(question)
            sources = [doc.metadata['source'] for doc in docs]
            return jsonify({'ragAnswer': answer, 
                            'sources':sources})
        
        else:
            curatedAns = sim_results[0].payload['answer']
            history.add_user_message(question)
            history.add_ai_message(curatedAns)
            return jsonify({'Id':sim_results[0].id,
                            'question':sim_results[0].payload['question'], 
                            'answer': sim_results[0].payload['answer']})

@app.route('/ingest', methods=['GET'])
def handle_ingestion():
    group = request.args.get('group')
    if group is None:
        return jsonify({"error": "Missing required query parameters"}), 400
    
    if group == "FAQ":
        try:
            faq_collection = CollectionUploadChecker(FAQ_COLLECTION_NAME, FAQ_LOC)
            faq_collection.check_collection_upload_data()
            response_data = {"message": f"Data ingested successfully with collection: {FAQ_COLLECTION_NAME}"}
            return jsonify(response_data)
        
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

    elif group == "docs" or group == "website":
        try:
            read_docs = PrepareVectorDB(DATA_LOC)
            read_docs.prepare_and_save_vectordb(COLLECTION_NAME)
            response_data = {"message": f"Data ingested successfully with collection: {COLLECTION_NAME}"}
            return jsonify(response_data)
        
        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    logging.info("scraping is started!")
    urls = []

    if 'url' in request.form:
        urls.append(request.form['url'])

    elif 'file' in request.files:
        file = request.files['file']
        
        #======Store the files from the request============#
        file.save(os.path.join("data", file.filename))
        with open(os.path.join("data", file.filename), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            urls.extend(row['URL'] for row in reader)
    else:
        return jsonify({"error": "No URL or CSV file provided"}), 400
    
    #======Store the scraped files in the s3============#
    folder_path = os.path.join(os.getcwd(), 'docs') 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for url in urls:
        parsed_html, error = Scraper.scrape_website(url)
        if error:
            return jsonify({"error": str(error)}), 500

        parsed_url = urlparse(url)
        filename = parsed_url.path.strip('/').replace('/', '-') + ".md"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'w') as file:
            file.write(parsed_html.cleaned_text)
    return jsonify({"message": "Scraping completed successfully"}), 200

@app.route('/health', methods=['GET'])
def health():
    health_response = {"message": "Health check successful"}
    return jsonify(health_response)

if __name__ == '__main__':
    app.run(debug=True , port=5001, host='0.0.0.0')
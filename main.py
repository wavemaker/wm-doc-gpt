from rag import chain
from flask import Flask, request, jsonify
import traceback
from src.prepare_db import PrepareVectorDB

from src.config import (
                        DATA_LOC,
                        COLLECTION_NAME)

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_answer():
    try:
        data = request.json
        question = data.get('question')
        if question:
            answer = chain.invoke(question)
            return jsonify({'answer': answer})
        else:
            return jsonify({'error': 'Question field is missing'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing the request'})


@app.route('/ingest', methods=['GET'])
def handle_ingestion():
    read_docs = PrepareVectorDB(DATA_LOC)
    res = read_docs.prepare_and_save_vectordb(COLLECTION_NAME)
    
    response_data = {"message": "Data ingested successfully"}
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True , port=5000)


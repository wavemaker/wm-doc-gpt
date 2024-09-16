from langchain_together import ChatTogether
import os
import re
from langchain import PromptTemplate, LLMChain
from flask import jsonify
import json

class DocSummary:
    def __init__(self, model="meta-llama/Llama-3-70b-chat-hf", max_tokens=500):
        self.llm = ChatTogether(
            together_api_key=os.getenv('TOGETHER_API'),
            model=model,
            max_tokens=max_tokens
        )

        self.template = """
                            Generate a detailed and comprehensive summary from the provided context {data}. 
                            Provide an in-depth summary of the core features of the content, explaining each feature and its significance.
                            Describe the practical benefits and implications of the content, including how it improves user experience or functionality.
                            Highlight notable details or additional information that is essential for understanding the content.
                            Ensure that the summary dynamically adjusts to include relevant sections based on the content, providing a well-rounded overview of the material.
                            The response must be a valid JSON array with the following structure:
                            - **"summary"**: A single string combining the detailed summary of the content. Avoid nested keys and values—just provide the plain summary text.
                            - **"questions"**: A list of follow-up questions derived from the content, formatted as strings. Include exactly 3 questions.
                            
                            Your response should contain **ONLY** the JSON array and nothing else. Do not include any additional text or formatting outside the JSON.
                            LLM Response: Here is the summary and questions:
                            """

        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=['data']        
            )
        self.llm_chain = LLMChain(
            prompt=self.prompt,
            llm=self.llm
        )
    
    def generate_summary(self, data):
        try:
            response = self.llm_chain.run(data=data)            
            if not response:
                return jsonify({
                    "answer": "Error: Received an empty response from the LLM.",
                    "faq_id": "",
                    "follow_up_questions": [],
                    "intent": "",
                    "response_from": "Summary",
                    "sources": ""
                })
            
            response = re.sub(r'[\x00-\x1F\x7F]', '', response)
            
            try:
                parsed_json = json.loads(response)
            except json.JSONDecodeError:
                return jsonify({
                    "answer": "Error: Could not decode the response into valid JSON.",
                    "faq_id": "",
                    "follow_up_questions": [],
                    "intent": "",
                    "response_from": "Summary",
                    "sources": ""
                })
            
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                data = parsed_json[0]
                
                summary = data.get("summary", "")
                questions = data.get("questions", [])
                
                formatted_response = {
                    "answer": summary,
                    "faq_id": "",  
                    "follow_up_questions": questions,  
                    "intent": "",  
                    "response_from": "Summary",
                    "sources": ""  
                }
                
                return jsonify(formatted_response)
            else:
                return jsonify({
                    "answer": "",
                    "faq_id": "",
                    "follow_up_questions": [],
                    "intent": "",
                    "response_from": "Summary",
                    "sources": ""
                })

        except Exception as e:
            return jsonify({
                "answer": f"Error: {str(e)}",
                "faq_id": "",
                "follow_up_questions": [],
                "intent": "",
                "response_from": "Summary",
                "sources": ""
            })

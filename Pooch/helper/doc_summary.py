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

        self.template = """Generate a detailed and comprehensive summary from the provided context {data}. 
                        Provide an in-depth summary of the core features of the content, explaining each feature and its significance.
                        Describe the practical benefits and implications of the content, including how it improves user experience or functionality.
                        Highlight notable details or additional information that is essential for understanding the content.
                        Ensure that the summary dynamically adjusts to include relevant sections based on the content, providing a well-rounded overview of the material.
                        The response should be a valid JSON object with the following structure:
                        - **"summary"**: A single string combining the detailed summary of the Overview and Topics sections into one cohesive text. Avoid nested keys and values—just provide the plain summary text.
                        - **"questions"**: A list of follow-up questions derived from the content, formatted as strings and make sure list should have only 3 questions.
                        Ensure the response is structured as a valid JSON array.and **ONLY** return the JSON array.
                        Here is the summary and questions:
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
        response = self.llm_chain.run(data=data)
        print(response)
        response = re.sub(r'[\x00-\x1F\x7F]', '', response)
        parsed_json = json.loads(response)
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
from langchain_together import ChatTogether
import os
from langchain import PromptTemplate
from langchain import LLMChain
from src.bot.faq_chat import search

class FollowUpQuestionGenerator:
    def __init__(self, model="meta-llama/Llama-3-70b-chat-hf", max_tokens=500):
        self.llm = ChatTogether(
            together_api_key=os.getenv('TOGETHER_API'),
            model=model,
            max_tokens=max_tokens
        )
        self.template = """Imagine yourself as the user, adept at generating follow-up questions based on the provided questions {user_question}. 
                            Generally, users inquire about the WaveMaker low-code platform and its technical aspects and Try to generate the questions based on the type of intent. 
                            Aim to create 5 questions that will motivate the user to request a demo, with each question approximately 20 characters long.  
                            Make sure only return the questions in the list with double inverted commas are returned, separated by commas within square brackets without adding any sentences in the beginning.
                            """
        self.prompt = PromptTemplate(
            template=self.template,
            # input_variables=['user_question', 'FAQ_Questions']
            input_variables=['user_question']
        )
        self.llm_chain = LLMChain(
            prompt=self.prompt,
            llm=self.llm
        )
    
    def generate_followup_questions(self, user_question):
        # FAQ_Questions = search(user_question)
        # response = self.llm_chain.run(user_question = user_question, FAQ_Questions=FAQ_Questions)
        response = self.llm_chain.run(user_question = user_question)
        return response
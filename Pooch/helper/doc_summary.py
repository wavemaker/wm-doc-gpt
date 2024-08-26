from langchain_together import ChatTogether
import os
from langchain import PromptTemplate
from langchain import LLMChain

class DocSummary:
    def __init__(self, model="meta-llama/Llama-3-70b-chat-hf", max_tokens=500):
        self.llm = ChatTogether(
            together_api_key=os.getenv('TOGETHER_API'),
            model=model,
            max_tokens=max_tokens
        )
        # self.template = """Generate a concise and coherent summary from the given Context {data}. 
        #                     Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context.
        #                     Prioritize clarity and brevity while retaining the essential information. Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding. 
        #                     Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context. 
        #                     Provide context where necessary and avoid excessive technical jargon or verbosity.
        #                 The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging. around the 700 characters and Please identify worthwhile topics from the provided data that can be follow-ups and write 3 questions no longer than 20 words each return the reponse.
        #                 Make sure only return the JSON  as 
        #                 Summary : summary of data,
        #                 Followup questions : in the double inverted commas are returned, separated by commas within square brackets.
        #                 Make sure that without adding any sentences in the beginning otherwise i will.
        #                 ONLY return the JSON , I will get fired if you don't return JSON. Here is the output:,
        #                 """
#         self.template = """You are an AI assistant tasked with summarizing technical documents {data}. When provided with a URL to a document, you should read and analyze the content to extract the key points and present them in a concise, well-structured summary.
#                             Your summary should include the following sections:
#                             Overview: A brief introduction to the framework, including its purpose, target platforms, and core principles.
#                             Key Features: A bulleted list highlighting the most important capabilities, architecture, and integration points of the framework.
#                             Customization and Limitations: An explanation of the customization options available to developers, as well as any known limitations or unsupported features.
# Additional Considerations: Any other relevant information, such as themes, prefabs, or best practices for working with the framework.
# Always dont add these Overview, Key Features and Customization and Limitations, Additional Considerations make the changes based on the provided context.
# The summary should be written in clear, accessible language, avoiding jargon where possible. It should be well-organized and easy to read, with appropriate headings and formatting to make the key points stand out.
# When generating the summary, you may use up to 3 levels of headings (e.g., ## Header, ### Sub-Header). The tone should be informative and helpful, providing the reader with a solid understanding of the framework's capabilities and how it can be used in web development projects. and Please identify worthwhile topics from the provided data that can be follow-ups and write 3 questions no longer than 20 words each return the reponse.
#                         Make sure only return the JSON  as 
#                         Summary : summary of data,
#                         Followup questions : in the double inverted commas are returned, separated by commas within square brackets.
#                         Make sure that without adding any sentences in the beginning otherwise i will.
#                         ONLY return the JSON , I will get fired if you don't return JSON. Here is the output:,
#                         """
        self.template = """Please provide a summary of the provided {data}.
                            Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context.
                            Prioritize clarity and brevity while retaining the essential information. Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding. 
                            Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context. 
                            Provide context where necessary and avoid excessive technical jargon or verbosity.
                       The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging. around the 700 characters and Please identify worthwhile topics from the provided data that can be follow-ups and write 3 questions no longer than 20 words each return the reponse.
                        The summary should be concise, typically no more than a few paragraphs, while capturing the essential information a reader would need to understand the content of the webpage.
                        Please identify worthwhile topics from the provided data that can be follow-ups and write 3 questions no longer than 20 words each return the reponse.
                        
                        Make sure only return the JSON  as 
                        Summary : summary of data,
                        Followup questions : in the double inverted commas are returned, separated by commas within square brackets.
                        Make sure that without adding any sentences in the beginning otherwise i will.
                        ONLY return the JSON , I will get fired if you don't return JSON. Here is the output:,
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
        response = self.llm_chain.run(data = data)
        return response
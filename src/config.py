
#Data Location#
DATA_LOC = "/Users/chiranjeevib_500350/wavemaker/Project/MarketingRag/data/testData_copy"

#Qudarant Conf#
PORT = 6333
HOSTNAME = "localhost"
COLLECTION_NAME = "WAVE"
QUDRANT_URL = "http://localhost:6333"
PERSIST_DIRECTORY = ""


##LLM
MODEL = 'gpt-4-turbo-preview'
TEMPERATURE = 0

##System Message
SYSTEM_MSG = """
            <|system|>>
            You are an assistant for WaveMaker, drawing from the provided data. If you encounter a question to which you don't know the answer, simply respond with 'Sorry, I'm not familiar with this question. Please visit the 'https://www.wavemaker.com/' or refrain from adding any unnecessary information.
            CONTEXT: {context}
            </s>
            <|user|>
            {query}
            </s>
            <|assistant|>
            """
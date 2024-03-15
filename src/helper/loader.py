import json
import logging
import pandas as pd
from langchain.document_loaders import DirectoryLoader

class CustomDirectoryLoader:
    def __init__(self, directory, glob, loader_cls):
        self.directory = directory
        self.glob = glob
        self.loader_cls = loader_cls

    def load(self):
        try:
            loader = DirectoryLoader(self.directory, glob=self.glob, loader_cls=self.loader_cls)
            data = loader.load()
            return data
        except FileNotFoundError as e:
            logging.error(f"Directory not found: {self.directory}")
            return None
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None

# loader = CustomDirectoryLoader(DATA_LOC, glob="./*.md", loader_cls=TextLoader)
# data = loader.load()
# print(data)

class CustomFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            if self.file_path.endswith('.json'):
                with open(self.file_path, 'r') as file:
                    data = json.load(file)
                return data
            elif self.file_path.endswith('.csv'):
                with open(self.file_path, 'r') as file:
                    reader = pd.read_csv(file)
                return reader
        except FileNotFoundError as e:
            logging.error(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            return None


# read_data = CustomFileLoader(FAQ_LOC)
# data = read_data.load()
# print(data)



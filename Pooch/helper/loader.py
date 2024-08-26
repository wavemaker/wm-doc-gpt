import json
import os
from glob import glob
import logging
import pandas as pd
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
import nltk
nltk.download('punkt')

nltk.data.path.append('/Users/chiranjeevib_500350/nltk_data/')



class CustomDirectoryLoader:
    def __init__(self, directory, glob, loader_cls):
        self.directory = directory
        self.glob = glob
        self.loader_cls = loader_cls
    
    def load(self):
        try:
            if os.path.isfile(self.directory):
                loader = UnstructuredMarkdownLoader(self.directory)
                self.data = loader.load()
                logging.info(f"Loading {os.path.splitext(self.directory)[1]} file is done!")
            
            elif os.path.isdir(self.directory):
                loader = DirectoryLoader(self.directory, loader_cls=self.loader_cls)
                self.data = loader.load()
                if self.data:
                    logging.info("Loading .md documents from directory is done")
                else:
                    logging.warning("No .md documents found or loaded")
            else:
                raise ValueError(f"Path '{self.directory}' is neither a file nor a directory")
            return self.data
        except FileNotFoundError as e:
            logging.error(f"File or directory not found: {self.directory}")
            return None

        
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



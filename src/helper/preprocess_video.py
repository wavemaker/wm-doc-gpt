# import os
# import fitz 
# from PyPDF2 import PdfReader
# from flask import Flask, request, jsonify
# from src.config.config import files_
# import logging


# app = Flask(__name__)
# app.config.from_object(files_)

# class PDFProcessor:
#     def __init__(self, pdf_path):
#         self.pdf_path = pdf_path
        
#         try:
#             logging.debug(f"Initializing PDFProcessor with file: {pdf_path}")
#             self.reader = PdfReader(pdf_path)
#             self.document = fitz.open(pdf_path)
        
#         except Exception as e:
#             logging.error(f"Error opening PDF file {pdf_path}: {e}")
#             raise RuntimeError(f"Error opening PDF file {pdf_path}: {e}")
#         self.text_output_path = pdf_path.replace('.pdf', '.md')

#     def extract_text(self):
#         try:
#             logging.debug(f"Extracting text from PDF: {self.pdf_path}")
#             number_of_pages = len(self.reader.pages)
#             all_text = []
#             for i in range(number_of_pages):
#                 page = self.reader.pages[i]
#                 text = page.extract_text()
#                 all_text.append(text)
#             logging.info(f"Successfully extracted text from {number_of_pages} pages of PDF: {self.pdf_path}")
#             return '\n'.join(all_text)
        
#         except Exception as e:
#             logging.error(f"Error extracting text from {self.pdf_path}: {e}")
#             raise RuntimeError(f"Error extracting text from {self.pdf_path}: {e}")

#     def extract_specific_hyperlinks(self, base_uri):
#         specific_links = []
#         try:
#             logging.debug(f"Extracting specific hyperlinks from PDF: {self.pdf_path}")
#             for page_num in range(self.document.page_count):
#                 page = self.document.load_page(page_num)
#                 links = page.get_links()
#                 for link in links:
#                     if 'uri' in link:
#                         uri = link['uri']
#                         if isinstance(uri, str) and uri.startswith(base_uri):
#                             specific_links.append(uri)
#             logging.info(f"Successfully extracted {len(specific_links)} specific hyperlinks from PDF: {self.pdf_path}")
#             return specific_links
        
#         except Exception as e:
#             logging.error(f"Error extracting hyperlinks from {self.pdf_path}: {e}")
#             raise RuntimeError(f"Error extracting hyperlinks from {self.pdf_path}: {e}")

#     def save_to_md(self, text, links):
#         md_filename = None  # Initialize md_filename
#         try:
#             # Attempt to construct the markdown filename
#             md_output_folder = files_.MD_OUTPUT_FOLDER
#             if not md_output_folder:
#                 raise ValueError("MD_OUTPUT_FOLDER is not set in the configuration.")

#             md_filename = os.path.join(md_output_folder, os.path.basename(self.pdf_path).replace('.pdf', '.md'))
            
#             # Save the markdown content
#             with open(md_filename, 'w', encoding='utf-8') as f:
#                 f.write(text)
#                 f.write('\n\n# Hyperlinks\n')
#                 for link in links:
#                     f.write(f'- {link}\n')
            
#             return md_filename
        
#         except Exception as e:
#             raise RuntimeError(f"Error saving Markdown file to {md_filename if md_filename else 'unknown'}: {e}")


# def format_hyperlinks(hyperlinks):
#     return '\n'.join(hyperlinks)


import os
import fitz 
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from src.config.config import files_

app = Flask(__name__)
app.config.from_object(files_)

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
        try:
            self.reader = PdfReader(pdf_path)
            self.document = fitz.open(pdf_path)
        
        except Exception as e:
            raise RuntimeError(f"Error opening PDF file {pdf_path}: {e}")
        self.text_output_path = pdf_path.replace('.pdf', '.md')

    def extract_text(self):
        try:
            number_of_pages = len(self.reader.pages)
            all_text = []
            for i in range(number_of_pages):
                page = self.reader.pages[i]
                text = page.extract_text()
                all_text.append(text)
            return '\n'.join(all_text)
        
        except Exception as e:
            raise RuntimeError(f"Error extracting text from {self.pdf_path}: {e}")

    def extract_specific_hyperlinks(self, base_uri):
        specific_links = []
        try:
            for page_num in range(self.document.page_count):
                page = self.document.load_page(page_num)
                links = page.get_links()
                for link in links:
                    if 'uri' in link:
                        uri = link['uri']
                        if isinstance(uri, str) and uri.startswith(base_uri):
                            specific_links.append(uri)
            return specific_links
        
        except Exception as e:
            raise RuntimeError(f"Error extracting hyperlinks from {self.pdf_path}: {e}")

    def save_to_md(self, text, links):
        md_filename = None  # Initialize md_filename with None or a default value
        try:
            # Build the path for the Markdown file
            md_filename = os.path.join(files_.MD_OUTPUT_FOLDER, os.path.basename(self.pdf_path).replace('.pdf', '.md'))
            
            # Write the extracted text and hyperlinks to the Markdown file
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(text)
                f.write('\n\n# Hyperlinks\n')
                for link in links:
                    f.write(f'- {link}\n')
            
            return md_filename
        
        except Exception as e:
            error_msg = f"Error saving Markdown file to {md_filename if md_filename else 'unknown location'}: {e}"
            raise RuntimeError(error_msg)

def format_hyperlinks(hyperlinks):
    return '\n'.join(hyperlinks)
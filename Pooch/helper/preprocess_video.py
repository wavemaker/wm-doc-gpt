from PyPDF2 import PdfReader
import fitz
import os
import logging
import shutil
from flask import jsonify
from Pooch.helper.prepare_db import PrepareAndSaveVideoTranscribe


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
        try:
            self.reader = PdfReader(pdf_path)
            self.document = fitz.open(pdf_path)
        
        except Exception as e:
            raise RuntimeError(f"Error opening PDF file {pdf_path}: {e}")

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
        md_filename = None
        try:
            output_dir = 'videos_'
            md_filename = os.path.join(output_dir, os.path.basename(self.pdf_path).replace('.pdf', '.md'))
            
            if os.path.exists(md_filename):
                os.remove(md_filename)
            
            os.makedirs(os.path.dirname(md_filename), exist_ok=True)
            
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

def process_pdf_directory(VIDEO_SOURCES):
    results = []
    base_uri = 'https://app.guidde.com'

    # def ensure_directories():
    #     output_dir = 'Videos_'

    #     # if os.path.exists(output_dir):
    #     #     shutil.rmtree(output_dir)

    #     os.makedirs(output_dir, exist_ok=True)  

    # ensure_directories()

    try:
        for root, dirs, files in os.walk(VIDEO_SOURCES):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)                    
                    try:
                        pdf_processor = PDFProcessor(file_path)
                        
                        text = pdf_processor.extract_text() 
                        hyperlinks = pdf_processor.extract_specific_hyperlinks(base_uri)
                        
                        md_content = pdf_processor.save_to_md(text, hyperlinks)
                        
                        formatted_hyperlinks = format_hyperlinks(hyperlinks)
                        
                        results.append((file_path, formatted_hyperlinks, md_content))

                    except Exception as e:
                        logging.error(f"Error processing PDF file {file_path}: {e}")
                        results.append({'error': f"Error processing file {file_path}: {str(e)}"})

    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return results

def ingest_markdown_content(results, VIDEO_COLLECTION):
    error_messages = []
    for file_path, hyperlinks, md_content in results:
        if isinstance(hyperlinks, str) and isinstance(md_content, str):
            try:
                read_docs = PrepareAndSaveVideoTranscribe(md_content, VIDEO_COLLECTION)
                read_docs.prepare_and_save_transcribe_data(hyperlinks)

                if read_docs is None:
                    error_messages.append(f"PDF data ingestion failed with collection: {VIDEO_COLLECTION}")
            except Exception as e:
                logging.error(f"Error processing Markdown content: {e}")
                error_messages.append(f"Failed to process Markdown content: {str(e)}")

    if error_messages:
        return jsonify({'errors': error_messages}), 500
    else:
        return jsonify({'message': f'PDF transcription successfully ingested into: {VIDEO_COLLECTION}'}), 200
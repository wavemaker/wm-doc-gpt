import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dataclasses import dataclass
import pdfplumber
import io
import logging

UNWANTED = ["sticky", "hidden"]

@dataclass
class ParsedHTML:
    title: str | None
    cleaned_text: str

class Scraper:
    @staticmethod
    def scrape_website(url):
        try:
            response = requests.get(url)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else ''
            
            unwanted_classes = UNWANTED
            for undesired_element in unwanted_classes:
                [
                    tag.extract()
                    for tag in soup.find_all(
                        class_=lambda x: x and undesired_element in x.split()
                    )
                ]

            text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            extracted_text = ' '.join([p.get_text() for p in text_elements]).strip()
            return ParsedHTML(title=title, cleaned_text=extracted_text), None
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Error scraping website {url}: {e}")
            return None, f"Error scraping website {url}: {e}"
        

class ScrapePDFAndSave:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url
    
    def extract_filename_from_url(self):
        return os.path.basename(self.pdf_url)
    
    def read_pdf_from_web(self):
        try:
            response = requests.get(self.pdf_url)
            response.raise_for_status()  
            filename = self.extract_filename_from_url()
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            return True, text, filename
        
        except Exception as e:
            logging.error(f"Error while reading pdf from web{e}")
            return False, None, None
    
    def save_to_md(self, text, filename):
        try:
            md_filename = os.path.splitext(filename) + ".md"
            
            with open(md_filename, 'w') as f:
                f.write(text)
            logging.info(f"PDF data from web saved to {md_filename}")
            return True
        
        except Exception as e:
            logging.error(f"Error while saving pdf to .md {e}")
            return False
    
    def convert_to_md(self):
        try:
            success, pdf_text, pdf_filename = self.read_pdf_from_web()
            
            if success:
                if self.save_to_md(pdf_text, pdf_filename):
                    logging.info("Conversion successful from pdf to .md")
                    return True
            
            logging.error("Conversion failed while convertinf pdf to .md")
            return False
        
        except Exception as e:
            logging.error(f"Error while converting pdf to .md {e}")
            return False
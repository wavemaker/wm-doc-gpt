import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dataclasses import dataclass
import PyPDF2
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
        

def scrape_pdf_and_save(url, folder_path):
    try:
        response = requests.get(url)
        response.raise_for_status() 

        with open(os.path.join(folder_path, 'temp.pdf'), 'wb') as file:
            file.write(response.content)

        text_content = ''
        with open(os.path.join(folder_path, 'temp.pdf'), 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            for page_number in range(num_pages):
                page = pdf_reader.pages[page_number]
                text_content += page.extract_text()

        parsed_url = urlparse(url)
        file_name = parsed_url.path.strip('/')

        filename = os.path.join(parsed_url.netloc, file_name).replace('/', '-')
        file_path = os.path.join(folder_path, filename + '.md')
        with open(file_path, 'w') as file:
            file.write(text_content)

        return "Reading the data from .pdf is done"
    except Exception as e:
        return str(e)
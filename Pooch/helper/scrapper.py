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

class GitHubContentFetcher:
    def __init__(self, base_github_raw_url):
        self.base_github_raw_url = base_github_raw_url

    def convert_to_github_raw_url(self, doc_url):
        """
        Convert a WaveMaker documentation URL to a GitHub raw Markdown content URL.
        
        Parameters:
        doc_url (str): The URL of the WaveMaker documentation.
        
        Returns:
        str: The raw GitHub content URL.
        """
        try:
            logging.info(f"Converting URL: {doc_url}")
            relative_path = doc_url.replace("https://docs.wavemaker.com/learn/", "")
            relative_path = relative_path.rstrip('/')
            github_raw_url = f"{self.base_github_raw_url}/{relative_path}.md"
            logging.info(f"Converted to GitHub raw URL: {github_raw_url}")
            return github_raw_url
        except Exception as e:
            self.logger.error(f"Error converting URL: {e}")
            raise

    def fetch_content_from_github(self, doc_url):
        """
        Fetch the content of a WaveMaker documentation page by converting the URL to GitHub raw URL.
        
        Parameters:
        doc_url (str): The URL of the WaveMaker documentation.
        
        Returns:
        str: The content of the corresponding GitHub Markdown file.
        """
        try:
            github_raw_url = self.convert_to_github_raw_url(doc_url)
            logging.info(f"Fetching content from GitHub URL: {github_raw_url}")
            response = requests.get(github_raw_url)
            response.raise_for_status()
            content = response.text
            logging.info(f"Content fetched successfully from: {github_raw_url}")
            return content
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"Other error occurred: {err}")
        return None

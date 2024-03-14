import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dataclasses import dataclass
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
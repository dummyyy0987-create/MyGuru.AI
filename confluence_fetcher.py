import os
import pickle
import requests
from atlassian import Confluence
from PyPDF2 import PdfReader
from io import BytesIO
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenceDataFetcher:
    """Fetch and process data from Confluence including pages and PDFs"""
    
    def __init__(self, confluence_url: str, username: str, api_token: str):
        self.confluence = Confluence(
            url=confluence_url,
            username=username,
            password=api_token,
            cloud=True
        )
        self.confluence_url = confluence_url
        
    def get_all_spaces(self) -> List[Dict]:
        """Get all accessible Confluence spaces"""
        try:
            spaces = self.confluence.get_all_spaces(start=0, limit=100)
            return spaces.get('results', [])
        except Exception as e:
            logger.error(f"Error fetching spaces: {e}")
            return []
    
    def get_pages_in_space(self, space_key: str) -> List[Dict]:
        """Get all pages in a specific space"""
        try:
            pages = self.confluence.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=500,
                expand='body.storage,version'
            )
            return pages
        except Exception as e:
            logger.error(f"Error fetching pages from space {space_key}: {e}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict:
        """Get detailed content of a specific page"""
        try:
            page = self.confluence.get_page_by_id(
                page_id=page_id,
                expand='body.storage,version,ancestors'
            )
            return page
        except Exception as e:
            logger.error(f"Error fetching page {page_id}: {e}")
            return {}
    
    def extract_text_from_page(self, page: Dict) -> str:
        """Extract clean text from Confluence page HTML"""
        from html.parser import HTMLParser
        
        class HTMLTextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
            
            def handle_data(self, data):
                self.text.append(data)
            
            def get_text(self):
                return ' '.join(self.text)
        
        body = page.get('body', {}).get('storage', {}).get('value', '')
        parser = HTMLTextExtractor()
        parser.feed(body)
        return parser.get_text()
    
    def get_attachments(self, page_id: str) -> List[Dict]:
        """Get all attachments for a page"""
        try:
            attachments = self.confluence.get_attachments_from_content(page_id)
            return attachments.get('results', [])
        except Exception as e:
            logger.error(f"Error fetching attachments for page {page_id}: {e}")
            return []
    
    def download_pdf_content(self, attachment: Dict) -> str:
        """Download and extract text from PDF attachment"""
        try:
            download_link = attachment.get('_links', {}).get('download')
            if not download_link:
                return ""
            
            # Make it absolute URL
            if download_link.startswith('/'):
                download_link = self.confluence_url + download_link
            
            # Download PDF
            response = requests.get(
                download_link,
                auth=(self.confluence.username, self.confluence.password)
            )
            
            if response.status_code == 200:
                pdf_file = BytesIO(response.content)
                pdf_reader = PdfReader(pdf_file)
                
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text
            else:
                logger.error(f"Failed to download PDF: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""
    
    def fetch_all_content(self) -> List[Dict]:
        """Fetch all content from Confluence including pages and PDFs"""
        all_documents = []
        
        # Get all spaces
        spaces = self.get_all_spaces()
        logger.info(f"Found {len(spaces)} spaces")
        
        for space in spaces:
            space_key = space.get('key')
            space_name = space.get('name')
            logger.info(f"Processing space: {space_name} ({space_key})")
            
            # Get all pages in space
            pages = self.get_pages_in_space(space_key)
            
            for page in pages:
                page_id = page.get('id')
                page_title = page.get('title')
                
                # Get full page content
                full_page = self.get_page_content(page_id)
                page_text = self.extract_text_from_page(full_page)
                
                # Create document entry
                doc = {
                    'id': page_id,
                    'title': page_title,
                    'content': page_text,
                    'space': space_name,
                    'space_key': space_key,
                    'type': 'page',
                    'url': f"{self.confluence_url}/wiki/spaces/{space_key}/pages/{page_id}"
                }
                all_documents.append(doc)
                
                # Check for PDF attachments
                attachments = self.get_attachments(page_id)
                for attachment in attachments:
                    if attachment.get('title', '').lower().endswith('.pdf'):
                        logger.info(f"Processing PDF: {attachment.get('title')}")
                        pdf_text = self.download_pdf_content(attachment)
                        
                        if pdf_text:
                            pdf_doc = {
                                'id': attachment.get('id'),
                                'title': f"{page_title} - {attachment.get('title')}",
                                'content': pdf_text,
                                'space': space_name,
                                'space_key': space_key,
                                'type': 'pdf',
                                'parent_page': page_title,
                                'url': f"{self.confluence_url}/wiki{attachment.get('_links', {}).get('webui', '')}"
                            }
                            all_documents.append(pdf_doc)
        
        logger.info(f"Total documents fetched: {len(all_documents)}")
        return all_documents
    
    def save_documents(self, documents: List[Dict], filepath: str):
        """Save fetched documents to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(documents, f)
        logger.info(f"Saved {len(documents)} documents to {filepath}")
    
    def load_documents(self, filepath: str) -> List[Dict]:
        """Load documents from disk"""
        with open(filepath, 'rb') as f:
            documents = pickle.load(f)
        logger.info(f"Loaded {len(documents)} documents from {filepath}")
        return documents

import os
import pickle
from typing import List, Dict
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """LangChain FAISS vector store for semantic search on Confluence documents"""
    
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str, embedding_deployment: str):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=embedding_deployment,
            chunk_size=16  # For batch processing
        )
        self.vectorstore = None
        self.documents = []
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def create_index(self, documents: List[Dict]):
        """Create FAISS index from documents using LangChain"""
        logger.info("Creating FAISS index with LangChain...")
        
        langchain_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
            
            # Split document into chunks
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                # Create LangChain Document with metadata
                metadata = {
                    'doc_id': doc.get('id'),
                    'title': doc.get('title'),
                    'space': doc.get('space'),
                    'space_key': doc.get('space_key'),
                    'type': doc.get('type'),
                    'url': doc.get('url'),
                    'chunk_index': i,
                    'source': doc.get('url')
                }
                
                langchain_doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)
        
        logger.info(f"Created {len(langchain_docs)} document chunks from {len(documents)} documents")
        
        # Create FAISS vector store
        logger.info("Generating embeddings and building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=langchain_docs,
            embedding=self.embeddings
        )
        
        self.documents = documents
        logger.info(f"FAISS index created successfully")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for most relevant chunks using LangChain FAISS"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        # Use similarity search with score
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            result = {
                'text': doc.page_content,
                'title': doc.metadata.get('title'),
                'space': doc.metadata.get('space'),
                'type': doc.metadata.get('type'),
                'url': doc.metadata.get('url'),
                'distance': float(score),
                'relevance_score': 1 / (1 + float(score))
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_retriever(self, k: int = 5):
        """Get LangChain retriever for ConversationalRetrievalChain"""
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return None
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def save_index(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(os.path.dirname(index_path))
        
        # Save metadata
        metadata = {
            'documents': self.documents
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk"""
        index_dir = os.path.dirname(index_path)
        
        if not os.path.exists(index_dir) or not os.path.exists(metadata_path):
            logger.error("Index or metadata files not found")
            return False
        
        # Load FAISS index
        self.vectorstore = FAISS.load_local(
            index_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.documents = metadata['documents']
        
        logger.info(f"Index loaded successfully")
        return True


def extract_github_urls(text: str) -> List[str]:
    """Extract GitHub repository URLs from text"""
    github_pattern = r'https?://github\.com/[\w\-]+/[\w\-.]+'
    urls = re.findall(github_pattern, text)
    return list(set(urls))  # Remove duplicates

"""
Setup script to fetch Confluence data and create FAISS index
Run this script once to index your Confluence documentation
"""

from config import Config
from confluence_fetcher import ConfluenceDataFetcher
from vector_store import VectorStore
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main setup function"""
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        Config.validate()
        
        # Initialize Confluence fetcher
        logger.info("Initializing Confluence fetcher...")
        fetcher = ConfluenceDataFetcher(
            confluence_url=Config.CONFLUENCE_URL,
            username=Config.CONFLUENCE_USERNAME,
            api_token=Config.CONFLUENCE_API_TOKEN
        )
        
        # Fetch all documents
        logger.info("Fetching Confluence documents (this may take a while)...")
        documents = fetcher.fetch_all_content()
        
        if not documents:
            logger.error("No documents fetched. Please check your Confluence configuration.")
            return
        
        # Save documents
        docs_path = os.path.join(Config.VECTOR_STORE_PATH, 'documents.pkl')
        fetcher.save_documents(documents, docs_path)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            embedding_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        # Create FAISS index
        logger.info("Creating FAISS index (this will take some time)...")
        vector_store.create_index(documents)
        
        # Save index
        logger.info("Saving FAISS index...")
        vector_store.save_index(Config.FAISS_INDEX_PATH, Config.METADATA_PATH)
        
        logger.info("✅ Setup completed successfully!")
        logger.info(f"Indexed {len(documents)} documents")
        logger.info("\nYou can now run the chatbot with: streamlit run app.py")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()

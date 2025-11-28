import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Azure OpenAI and Confluence"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    # Confluence Configuration
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
    
    # GitHub Configuration (Optional)
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    GITHUB_ORGANIZATION = os.getenv("GITHUB_ORGANIZATION")  # Optional: limit to specific org
    
    # Vector Store Configuration
    VECTOR_STORE_PATH = "vector_store"
    FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "confluence_index.faiss")
    METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        required_vars = [
            ("AZURE_OPENAI_API_KEY", cls.AZURE_OPENAI_API_KEY),
            ("AZURE_OPENAI_ENDPOINT", cls.AZURE_OPENAI_ENDPOINT),
            ("CONFLUENCE_URL", cls.CONFLUENCE_URL),
            ("CONFLUENCE_USERNAME", cls.CONFLUENCE_USERNAME),
            ("CONFLUENCE_API_TOKEN", cls.CONFLUENCE_API_TOKEN),
        ]
        
        missing = [name for name, value in required_vars if not value]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return True

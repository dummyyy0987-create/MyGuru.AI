import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from config import Config
from vector_store import VectorStore, extract_github_urls
from github_search import GitHubSearcher
import base64
import os

# Optional database import
try:
    from database_search import DatabaseSearcher
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    DatabaseSearcher = None

from agent_tools import create_confluence_tool, create_github_tool
try:
    from agent_tools import create_database_tool
except ImportError:
    create_database_tool = None

from agents import create_confluence_agent, create_github_agent, create_supervisor_agent
try:
    from agents import create_database_agent
except ImportError:
    create_database_agent = None

import os
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Guru AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Clean minimal CSS
st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide hamburger menu */
    button[kind="header"] {
        display: none !important;
    }
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Full width container with padding for fixed header and bottom input */
    .block-container {
        max-width: 100% !important;
        padding: 70px 2rem 120px 2rem !important;
    }
    
    /* Main content area */
    .st-emotion-cache-1cei9z1 {
        padding-top: 70px !important;
        padding-bottom: 120px !important;
    }
    
    /* Fixed header at top */
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 0.5rem 0 0.5rem 0;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Chat messages - full width content area */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem !important;
        margin: 0.5rem 0;
        max-width: 100% !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
    }
    
    /* Target specific emotion-cache classes for chat messages */
    .st-emotion-cache-1iitq1e {
        padding: 1rem !important;
        gap: 0.75rem !important;
        align-items: center !important;
    }
    
    .st-emotion-cache-wfksaw {
        padding-right: 0rem !important;
        margin-right: 0rem !important;
    }
    
    .st-emotion-cache-58vgod {
        padding-right: 0rem !important;
        margin-right: 0rem !important;
    }
    
    /* Chat message avatar - center vertically */
    .stChatMessage [data-testid="chatAvatarIcon"] {
        margin: 0 !important;
        padding: 0 !important;
        flex-shrink: 0 !important;
        align-self: center !important;
    }
    
    /* Chat message content container - align with icon */
    [data-testid="stChatMessageContent"] {
        max-width: 100% !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        flex-grow: 1 !important;
    }
    
    /* Aggressively remove all right spacing and increase font size */
    .stChatMessage p,
    .stChatMessage span,
    .stChatMessage div,
    .stChatMessage * {
        color: #ffffff !important;
        margin: 0.25rem 0 !important;
        padding: 0 !important;
        padding-right: 0 !important;
        margin-right: 0 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
            /* Style links to be visible */
    .stChatMessage a,
    .stChatMessage a:link,
    .stChatMessage a:visited {
        color: #60a5fa !important;
        text-decoration: underline !important;
    }
    
    .stChatMessage a:hover {
        color: #93c5fd !important;
    }
        /* Style inline code and code blocks to be visible */
    .stChatMessage code,
    .stChatMessage pre,
    .stChatMessage pre code {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #fbbf24 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 0.95rem !important;
    }
    
    .stChatMessage pre {
        padding: 0.75rem !important;
        overflow-x: auto !important;
    }
    
    .stChatMessage pre code {
        padding: 0 !important;
        background-color: transparent !important;
    }
        
    
    /* Fix bullet list spacing */
    .stChatMessage ol,
    .stChatMessage ul {
        margin-left: 0 !important;
        padding-left: 1.5rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stChatMessage li {
        margin: 0.25rem 0 !important;
        padding-left: 0.25rem !important;
    }
    
    /* Target markdown containers */
    .stChatMessage [data-testid="stMarkdownContainer"],
    .stChatMessage .st-emotion-cache-vciuws {
        padding: 0 !important;
        padding-right: 0 !important;
        margin-right: 0 !important;
        max-width: 100% !important;
    }
    
    /* Target vertical blocks in chat */
    .stChatMessage .stVerticalBlock,
    .stChatMessage .st-emotion-cache-tn0cau {
        padding-right: 0 !important;
        margin-right: 0 !important;
        gap: 0 !important;
    }
    
    /* Target element containers */
    .stChatMessage .stElementContainer,
    .stChatMessage .st-emotion-cache-1vo6xi6 {
        padding-right: 0 !important;
        margin-right: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Remove extra padding from markdown elements in chat */
    .stChatMessage .st-emotion-cache-1v0mbdj,
    .stChatMessage .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Chat input styling - centered and narrower */
    .stChatInputContainer {
        background-color: transparent !important;
        padding: 1rem 0;
        max-width: 900px !important;
        margin: 0 auto !important;
        width: 100% !important;
    }
    
    .stChatInputContainer > div {
        background: rgba(80, 90, 130, 0.8) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
    }
    
    /* Center the input within footer */
    .st-emotion-cache-1vo6xi6 {
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="base-input"] > div {
        background-color: transparent !important;
        border: none !important;
    }
    
    div[data-baseweb="textarea"] textarea {
        background-color: transparent !important;
        color: #ffffff !important;
        font-size: 15px !important;
        caret-color: #ffffff !important;
    }
    
    div[data-baseweb="textarea"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Also target input fields for cursor color */
    input[type="text"],
    textarea,
    .stChatInputContainer input,
    .stChatInputContainer textarea {
        caret-color: #ffffff !important;
    }
    
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div {
        background-color: transparent !important;
    }
    
    /* Footer */
    footer {
        background-color: transparent !important;
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Solid footer background - full width */
    [data-testid="stBottomBlockContainer"] {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%) !important;
        padding-top: 1rem !important;
    }
    
    /* Footer containers with solid background and full width */
    .st-emotion-cache-i12q1z {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%) !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    .st-emotion-cache-6shykm {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%) !important;
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    .st-emotion-cache-1p2n2i4 {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%) !important;
        max-width: 100% !important;
        width: 100% !important;
    }
    
    /* Header styling */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_llm():
    """Initialize Azure OpenAI LLM"""
    try:
        Config.validate()
        llm = AzureChatOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            azure_deployment=Config.AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.7,
            max_tokens=1000
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None


def initialize_vector_store():
    """Initialize vector store"""
    try:
        vector_store = VectorStore(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            embedding_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        if os.path.exists(Config.FAISS_INDEX_PATH):
            vector_store.load_index(Config.FAISS_INDEX_PATH, Config.METADATA_PATH)
            return vector_store
        else:
            st.warning("⚠️ Vector store not found. Please run `python setup_index.py` first.")
            return None
            
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None


def initialize_agents(llm, vector_store, github_searcher, database_searcher):
    """Initialize multi-agent system"""
    try:
        confluence_tool = create_confluence_tool(vector_store)
        
        confluence_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        github_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        database_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        supervisor_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        confluence_agent = create_confluence_agent(llm, confluence_tool, confluence_memory)
        
        github_agent = None
        if github_searcher:
            github_tool = create_github_tool(github_searcher)
            github_agent = create_github_agent(llm, github_tool, github_memory)
        
        database_agent = None
        if database_searcher and create_database_tool and create_database_agent:
            database_tool = create_database_tool(database_searcher)
            database_agent = create_database_agent(llm, database_tool, database_memory)
        
        supervisor = create_supervisor_agent(llm, confluence_agent, github_agent, database_agent, supervisor_memory)
        
        return {
            'supervisor': supervisor,
            'confluence_agent': confluence_agent,
            'github_agent': github_agent,
            'database_agent': database_agent,
            'memories': {
                'confluence': confluence_memory,
                'github': github_memory,
                'database': database_memory,
                'supervisor': supervisor_memory
            }
        }
        
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        return None


def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logger.warning(f"Could not load image {image_path}: {e}")
        return None


def main():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = initialize_vector_store()
    
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    # Initialize GitHub searcher
    if 'github_searcher' not in st.session_state:
        github_token = Config.GITHUB_TOKEN
        github_org = Config.GITHUB_ORGANIZATION
        
        if github_token:
            try:
                st.session_state.github_searcher = GitHubSearcher(github_token, github_org)
                logger.info("GitHub searcher initialized")
            except Exception as e:
                logger.warning(f"GitHub initialization failed: {e}")
                st.session_state.github_searcher = None
        else:
            st.session_state.github_searcher = None
    
    # Initialize Database searcher
    if 'database_searcher' not in st.session_state:
        if DATABASE_AVAILABLE:
            db_server = Config.AZURE_SQL_SERVER
            db_database = Config.AZURE_SQL_DATABASE
            db_username = Config.AZURE_SQL_USERNAME
            db_password = Config.AZURE_SQL_PASSWORD
            
            if db_server and db_database and db_username and db_password:
                try:
                    st.session_state.database_searcher = DatabaseSearcher(
                        server=db_server,
                        database=db_database,
                        username=db_username,
                        password=db_password
                    )
                    logger.info("Database searcher initialized")
                except Exception as e:
                    logger.warning(f"Database initialization failed: {e}")
                    st.session_state.database_searcher = None
            else:
                st.session_state.database_searcher = None
        else:
            st.session_state.database_searcher = None
    
    # Initialize agents
    if 'agents' not in st.session_state and st.session_state.vector_store and st.session_state.llm:
        st.session_state.agents = initialize_agents(
            st.session_state.llm,
            st.session_state.vector_store,
            st.session_state.github_searcher,
            st.session_state.database_searcher
        )
    
    # Check initialization
    if not st.session_state.llm or not st.session_state.vector_store or not st.session_state.get('agents'):
        st.error("❌ Failed to initialize. Please check your configuration.")
        st.stop()
    
    # Fixed header at top
    # Check if logo image exists
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    logo_html = ""
    if os.path.exists(logo_path):
        logo_base64 = get_image_base64(logo_path)
        if logo_base64:
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" alt="Logo" style="width: 35px; height: 35px; object-fit: contain;" />'
    else:
        logo_html = "🧠"  # Fallback to emoji if no image
    
    st.markdown(f"""
        <div class="fixed-header">
            <h1 style="color: #ffffff; font-size: 1.75rem; font-weight: 600; margin: 0; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                {logo_html}
                Guru AI
            </h1>
            <p style="color: rgba(255, 255, 255, 0.6); margin-top: 0.25rem; font-size: 0.875rem;">
                Your intelligent assistant powered by Confluence, GitHub & Database
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display chat messages using Streamlit's native chat
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input - automatically stays at bottom
    if prompt := st.chat_input("Send a message..."):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    supervisor = st.session_state.agents['supervisor']
                    result = asyncio.run(supervisor(prompt))
                    response = result.get('answer', 'No answer generated.')
                except Exception as e:
                    logger.error(f"Error: {e}")
                    response = f"Error: {str(e)}"
            
            # Display the final response
            message_placeholder.markdown(response)
            
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

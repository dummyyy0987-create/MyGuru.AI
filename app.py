import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from config import Config
from vector_store import VectorStore, extract_github_urls
from github_search import GitHubSearcher

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
    page_title="AI Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Minimal CSS - only dark theme colors
st.markdown("""
<style>
    .stApp {
        background-color: #343541;
    }
    
    .stChatMessage {
        background-color: #444654;
    }
    
    /* Sidebar dark theme */
    [data-testid="stSidebar"] {
        background-color: #2c2c3a !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li {
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton button {
        background-color: #2c2c3a !important;
        color: #ffffff !important;
        border: 1px solid #4a4a5a !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #3a3a4a !important;
        border: 1px solid #5a5a6a !important;
    }
    
    /* Chat input styling - target all nested elements */
    .stChatInputContainer {
        background-color: #1a1a2e !important;
    }
    
    .stChatInputContainer > div {
        background-color: #3a3a4a !important;
    }
    
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"],
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="base-input"] > div {
        background-color: #3a3a4a !important;
    }
    
    /* Text area input field */
    div[data-baseweb="textarea"] textarea {
        background-color: #3a3a4a !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #a0a0a0 !important;
    }
    
    /* Remove any border or padding that might show white */
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        border: none !important;
    }
    
    /* Target the inner div that contains the textarea */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div {
        background-color: #3a3a4a !important;
    }
    
    /* Footer styling */
    footer {
        background-color: #343541 !important;
        color: #ffffff !important;
    }
    
    footer * {
        color: #ffffff !important;
    }
    
    /* Bottom container with input field */
    [data-testid="stBottomBlockContainer"] {
        background-color: #343541 !important;
    }
    
    .st-emotion-cache-i12q1z {
        background-color: #343541 !important;
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
    
    # Sidebar
    with st.sidebar:
        st.title("💬 AI Assistant")
        st.divider()
        
        st.markdown("### 📚 Data Sources")
        st.markdown("- Confluence")
        st.markdown("- GitHub")
        st.markdown("- Database")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.get('agents'):
                for memory in st.session_state.agents['memories'].values():
                    memory.clear()
            st.rerun()
        
        st.divider()
        st.caption(f"💬 {len(st.session_state.messages)//2} conversations")
    
    # Display chat messages using Streamlit's native chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input - automatically stays at bottom
    if prompt := st.chat_input("Send a message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    supervisor = st.session_state.agents['supervisor']
                    result = asyncio.run(supervisor(prompt))
                    response = result.get('answer', 'No answer generated.')
                except Exception as e:
                    logger.error(f"Error: {e}")
                    response = f"Error: {str(e)}"
            
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

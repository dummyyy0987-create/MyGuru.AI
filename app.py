import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
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

from agent_tools import create_github_tool,create_confluence_tool
try:
    from agent_tools import create_database_tool
except ImportError:
    create_database_tool = None

from agents import create_confluence_agent,create_github_agent, create_supervisor_agent
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
    page_title="MyGuru AI",
    page_icon="🧘‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ==========================
# AI BACKGROUND IMAGE WITH DARK OVERLAY
# ==========================
st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(
            rgba(0, 0, 0, 0.7), 
            rgba(0, 0, 0, 0.7)
        ),
        url("https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=1920");
        background-size: cover;
        background-position: center;
    }
    
    .stMarkdown, .stChatMessage {
        color: #ffffff !important;
    }

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
    
    div[data-baseweb="textarea"] textarea {
        background-color: #3a3a4a !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="textarea"] textarea::placeholder {
        color: #a0a0a0 !important;
    }
    
    div[data-baseweb="textarea"],
    div[data-baseweb="base-input"] {
        border: none !important;
    }
    
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInput"] > div > div > div {
        background-color: #3a3a4a !important;
    }
    
    footer {
        background-color: #343541 !important;
        color: #ffffff !important;
    }
    
    footer * {
        color: #ffffff !important;
    }
    
    [data-testid="stBottomBlockContainer"] {
        background-color: #343541 !important;
    }

    .title {
        font-size: 36px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
    }
    
    .caption {
        font-size: 18px;
        color: #ffffff;
        text-align: center;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
#       INITIALIZERS
# ==============================

def initialize_llm():
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
    try:
        confluence_tool = create_confluence_tool(vector_store)
        # Removed all Confluence tools/agents
        github_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        database_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        supervisor_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        confluence_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        confluence_agent = create_confluence_agent(llm,confluence_tool, confluence_memory)
        # GitHub Agent
        github_agent = None
        if github_searcher:
            github_tool = create_github_tool(github_searcher)
            github_agent = create_github_agent(llm, github_tool, github_memory)

        # Database Agent
        database_agent = None
        if database_searcher and create_database_tool and create_database_agent:
            database_tool = create_database_tool(database_searcher)
            database_agent = create_database_agent(llm, database_tool, database_memory)

        # Supervisor Agent (Confluence + GitHub + Database)
        supervisor = create_supervisor_agent(
            llm,
            confluence_agent,
            github_agent,
            database_agent,
            supervisor_memory
        )

        return {
            'supervisor': supervisor,
            'github_agent': github_agent,
            'database_agent': database_agent,
            'memories': {
                'github': github_memory,
                'database': database_memory,
                'supervisor': supervisor_memory
            }
        }
        
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        return None


# ==============================
#              MAIN
# ==============================

def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = initialize_vector_store()

    # GitHub Searcher
    if 'github_searcher' not in st.session_state:
        github_token = Config.GITHUB_TOKEN
        github_org = Config.GITHUB_ORGANIZATION
        
        if github_token:
            try:
                st.session_state.github_searcher = GitHubSearcher(github_token, github_org)
                logger.info("GitHub searcher initialized")
            except Exception as e:
                logger.warning(f"GitHub init failed: {e}")
                st.session_state.github_searcher = None
        else:
            st.session_state.github_searcher = None

    # Database Searcher
    if 'database_searcher' not in st.session_state:
        if DATABASE_AVAILABLE:
            db_server = Config.AZURE_SQL_SERVER
            db_database = Config.AZURE_SQL_DATABASE
            db_username = Config.AZURE_SQL_USERNAME
            db_password = Config.AZURE_SQL_PASSWORD
            
            if all([db_server, db_database, db_username, db_password]):
                try:
                    st.session_state.database_searcher = DatabaseSearcher(
                        server=db_server,
                        database=db_database,
                        username=db_username,
                        password=db_password
                    )
                except Exception as e:
                    st.session_state.database_searcher = None
            else:
                st.session_state.database_searcher = None
        else:
            st.session_state.database_searcher = None

    # Agents
    if 'agents' not in st.session_state and st.session_state.vector_store and st.session_state.llm:
        st.session_state.agents = initialize_agents(
            st.session_state.llm,
            st.session_state.vector_store,
            st.session_state.github_searcher,
            st.session_state.database_searcher
        )
    
    if not st.session_state.llm or not st.session_state.vector_store or not st.session_state.get('agents'):
        st.error("❌ Initialization failed. Check config.")
        st.stop()

    # Header
    st.markdown('<div class="title">MyGuru AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">One stop solution</div>', unsafe_allow_html=True)

    # Chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    if prompt := st.chat_input("Send a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    supervisor = st.session_state.agents['supervisor']
                    result = asyncio.run(supervisor(prompt))
                    response = result.get('answer', 'No answer generated.')
                except Exception as e:
                    response = f"Error: {str(e)}"
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()

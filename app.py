import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from config import Config
from vector_store import VectorStore, extract_github_urls
from github_search import GitHubSearcher
from agent_tools import create_confluence_tool, create_github_tool
from agents import create_confluence_agent, create_github_agent, create_supervisor_agent
import os
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Hide deploy button and hamburger menu */
    [data-testid="stToolbar"] {
        display: none;
    }
    
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    .stTextInput input {
        color: #000000 !important;
    }
    input[type="text"] {
        color: #000000 !important;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #1a1a1a;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #0d47a1;
    }
    .bot-message {
        background-color: #ffffff;
        border-left: 4px solid #4caf50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        color: #212121;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        border-left: 3px solid #ff9800;
        color: #424242;
    }
    .source-card strong {
        color: #1a1a1a;
    }
    .source-card small {
        color: #616161;
    }
    .source-card a {
        color: #1976d2;
        text-decoration: none;
        font-weight: 500;
    }
    .source-card a:hover {
        text-decoration: underline;
    }
    .github-link {
        background-color: #24292e;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
        display: inline-block;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_conversational_chain():
    """Initialize LangChain ConversationalRetrievalChain"""
    try:
        Config.validate()
        
        # Initialize Azure Chat OpenAI
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
        st.error(f"Error initializing Azure OpenAI LLM: {e}")
        return None


def initialize_vector_store():
    """Initialize and load vector store"""
    try:
        vector_store = VectorStore(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            embedding_deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        # Check if index exists
        if os.path.exists(Config.FAISS_INDEX_PATH):
            vector_store.load_index(Config.FAISS_INDEX_PATH, Config.METADATA_PATH)
            return vector_store
        else:
            st.warning("⚠️ Vector store not found. Please run `python setup_index.py` first to index your Confluence data.")
            return None
            
    except Exception as e:
        st.error(f"Error initializing vector store: {e}")
        return None


def initialize_agents(llm, vector_store, github_searcher):
    """Initialize multi-agent system"""
    try:
        # Create tools
        confluence_tool = create_confluence_tool(vector_store)
        
        # Create memories for each agent
        confluence_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        github_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        supervisor_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agents
        confluence_agent = create_confluence_agent(llm, confluence_tool, confluence_memory)
        
        github_agent = None
        if github_searcher:
            github_tool = create_github_tool(github_searcher)
            github_agent = create_github_agent(llm, github_tool, github_memory)
        
        # Create supervisor
        supervisor = create_supervisor_agent(llm, confluence_agent, github_agent, supervisor_memory)
        
        return {
            'supervisor': supervisor,
            'confluence_agent': confluence_agent,
            'github_agent': github_agent,
            'memories': {
                'confluence': confluence_memory,
                'github': github_memory,
                'supervisor': supervisor_memory
            }
        }
        
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        return None


def main():
    # Header
    st.title("🤖 AI Assistant")
    st.markdown("Ask me anything - I can search Confluence docs, GitHub repos, and databases")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize LLM
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_conversational_chain()
    
    # Initialize vector store
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = initialize_vector_store()
    
    # Initialize conversation memory
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # Initialize GitHub searcher
    if 'github_searcher' not in st.session_state:
        github_token = Config.GITHUB_TOKEN
        github_org = Config.GITHUB_ORGANIZATION
        
        if github_token:
            try:
                st.session_state.github_searcher = GitHubSearcher(github_token, github_org)
                logger.info("GitHub searcher initialized successfully")
            except Exception as e:
                logger.warning(f"GitHub searcher initialization failed: {e}")
                st.session_state.github_searcher = None
        else:
            logger.info("No GitHub token provided - GitHub search disabled")
            st.session_state.github_searcher = None
    
    # Initialize multi-agent system
    if 'agents' not in st.session_state and st.session_state.vector_store and st.session_state.llm:
        st.session_state.agents = initialize_agents(
            st.session_state.llm,
            st.session_state.vector_store,
            st.session_state.github_searcher
        )
    
    # Check if initialization was successful
    if not st.session_state.llm or not st.session_state.vector_store or not st.session_state.get('agents'):
        st.error("❌ Failed to initialize. Please check your configuration.")
        st.stop()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if 'sources' in message and message['sources']:
                    st.markdown("**📚 Confluence Sources:**")
                    for source in message['sources'][:3]:  # Show top 3 sources
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{source['title']}</strong> ({source['type']}) - {source['space']}<br>
                            <small>Relevance: {source['relevance_score']:.2%}</small><br>
                            <a href="{source['url']}" target="_blank">View in Confluence →</a>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display GitHub repositories if available
                if 'github_repos' in message and message['github_repos']:
                    st.markdown("**🔗 Relevant GitHub Repositories:**")
                    for repo in message['github_repos']:
                        st.markdown(f"""
                        <div class="source-card" style="border-left: 3px solid #24292e;">
                            <strong>⭐ {repo['name']}</strong> ({repo['stars']} stars)<br>
                            <p>{repo['description']}</p>
                            <small>Language: {repo['language']} | Topics: {', '.join(repo['topics'][:5])}</small><br>
                            <a href="{repo['url']}" target="_blank" style="color: #1976d2;">View Repository →</a>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display additional GitHub links if available
                if 'github_links' in message and message['github_links']:
                    # Filter out repos already shown
                    shown_repos = [repo['url'] for repo in message.get('github_repos', [])]
                    additional_links = [link for link in message['github_links'] if link not in shown_repos]
                    
                    if additional_links:
                        st.markdown("**🔗 Additional GitHub Links:**")
                        for link in additional_links:
                            st.markdown(f"- [{link}]({link})")
    
    # Input section at the bottom
    st.markdown("---")
    
    # Query input
    st.markdown('<p style="font-size: 14px; color: #424242; margin-bottom: 5px;">Ask your question:</p>', unsafe_allow_html=True)
    with st.form(key='query_form', clear_on_submit=True):
        query = st.text_input(
            "Query",
            placeholder="e.g., How do I configure the authentication service?",
            label_visibility="collapsed",
            key="user_query_input"
        )
        submit_button = st.form_submit_button("🚀 Send", use_container_width=True)
    
    # Process query
    if submit_button and query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query
        })
        
        # Show loading spinner
        with st.spinner("🤖 AI Agents are searching..."):
            try:
                # Use supervisor agent to coordinate search
                supervisor = st.session_state.agents['supervisor']
                
                # Run supervisor (it will coordinate Confluence and GitHub agents)
                result = asyncio.run(supervisor(query))
                
                response = result.get('answer', 'No answer generated.')
                confluence_used = result.get('confluence_used', False)
                github_used = result.get('github_used', False)
                
                # Extract sources and GitHub links from response
                sources = []
                github_repos = []
                github_links = extract_github_urls(response)
                
                # Add agent info to response
                agent_info = []
                if confluence_used:
                    agent_info.append("📚 Confluence Agent")
                if github_used:
                    agent_info.append("🐙 GitHub Agent")
                
                if agent_info:
                    response = f"*Consulted: {', '.join(agent_info)}*\n\n{response}"
                
            except Exception as e:
                logger.error(f"Error with agent system: {e}")
                response = f"Error generating response: {str(e)}"
                sources = []
                github_links = []
                github_repos = []
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': sources if 'sources' in locals() else [],
                'github_links': github_links if 'github_links' in locals() else [],
                'github_repos': github_repos if 'github_repos' in locals() else []
            })
        
        # Rerun to update the display
        st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        Your intelligent assistant with access to:
        
        📚 **Confluence** - Documentation & PDFs
        
        🐙 **GitHub** - Repositories & Code
        
        💾 **Database** - Structured data
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            # Clear all agent memories
            if st.session_state.get('agents'):
                for memory in st.session_state.agents['memories'].values():
                    memory.clear()
            st.rerun()


if __name__ == "__main__":
    main()

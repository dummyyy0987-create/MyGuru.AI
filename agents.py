from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import List
import logging

logger = logging.getLogger(__name__)


def create_confluence_agent(llm: AzureChatOpenAI, confluence_tool, memory: ConversationBufferMemory):
    """
    Create Confluence search agent
    
    Args:
        llm: Azure OpenAI LLM instance
        confluence_tool: Confluence search tool
        memory: Conversation memory
        
    Returns:
        AgentExecutor for Confluence searches
    """
    
    # Define prompt for Confluence agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Confluence Documentation Expert.

Your role: Extract and provide DIRECT, CONCISE answers from Confluence documentation.

Rules:
- Use confluence_search tool to find relevant docs
- READ the content and extract the EXACT answer
- Be brief and to the point - no extra explanations
- Include GitHub repo links if found in the documentation
- Cite sources at the end (title and URL)
- If no relevant info found, say: "No relevant information found in Confluence."

Format:
[Direct answer from the content]

Sources:
- [Document Title] (URL)
- [GitHub repos if any]
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, [confluence_tool], prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[confluence_tool],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor


def create_github_agent(llm: AzureChatOpenAI, github_tool, memory: ConversationBufferMemory):
    """
    Create GitHub search agent
    
    Args:
        llm: Azure OpenAI LLM instance
        github_tool: GitHub search tool
        memory: Conversation memory
        
    Returns:
        AgentExecutor for GitHub searches
    """
    
    # Define prompt for GitHub agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a GitHub Repository Expert Agent.

Your role is to search and retrieve information from accessible GitHub repositories.

Guidelines:
- Search GitHub repositories using the github_search tool
- Focus on README files and repository descriptions
- Provide repository details including stars, language, and topics
- Include direct links to relevant repositories
- Summarize README content when helpful
- Only search YOUR accessible repositories (private and public)

When answering:
1. Use the github_search tool to find relevant repositories
2. Extract key information from README files
3. Highlight the most relevant repositories
4. Provide direct GitHub URLs for users to explore
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, [github_tool], prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[github_tool],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor


def create_supervisor_agent(llm: AzureChatOpenAI, confluence_agent, github_agent, memory: ConversationBufferMemory):
    """
    Create supervisor agent that coordinates between Confluence and GitHub agents
    
    Args:
        llm: Azure OpenAI LLM instance
        confluence_agent: Confluence search agent executor
        github_agent: GitHub search agent executor
        memory: Conversation memory
        
    Returns:
        Supervisor function that routes queries
    """
    
    async def supervisor(query: str) -> dict:
        """
        Route query to appropriate agent(s) based on context
        
        Args:
            query: User query
            
        Returns:
            Dict with answer and sources
        """
        try:
            # First, try Confluence agent
            logger.info("Querying Confluence agent...")
            confluence_result = confluence_agent.invoke({"input": query})
            confluence_output = confluence_result.get('output', '')
            
            # Check if Confluence found sufficient information
            insufficient_phrases = [
                'no relevant information',
                'not found',
                'did not return',
                'no specific information',
                'could not find',
                'not seem related',
                'refine the search',
                'provide more details'
            ]
            
            is_insufficient = (
                any(phrase in confluence_output.lower() for phrase in insufficient_phrases) or
                len(confluence_output.strip()) < 100
            )
            
            logger.info(f"Confluence result length: {len(confluence_output)}, is_insufficient: {is_insufficient}")
            
            github_output = ""
            
            if is_insufficient:
                # Query GitHub agent as fallback
                logger.info("Confluence results insufficient. Querying GitHub agent...")
                github_result = github_agent.invoke({"input": query})
                github_output = github_result.get('output', '')
                logger.info(f"GitHub agent returned {len(github_output)} characters")
            
            # Combine results if both available
            if confluence_output and github_output:
                combined_answer = f"""Based on available sources:

**From Confluence Documentation:**
{confluence_output}

**From GitHub Repositories:**
{github_output}
"""
            elif confluence_output:
                combined_answer = confluence_output
            elif github_output:
                combined_answer = github_output
            else:
                combined_answer = "I couldn't find relevant information in either Confluence or GitHub repositories."
            
            return {
                'answer': combined_answer,
                'confluence_used': bool(confluence_output),
                'github_used': bool(github_output)
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor agent: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'confluence_used': False,
                'github_used': False
            }
    
    return supervisor

from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.memory import ConversationBufferMemory
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


def create_database_agent(llm: AzureChatOpenAI, database_tool, memory: ConversationBufferMemory):
    """
    Create Database search agent with text-to-SQL capability
    
    Args:
        llm: Azure OpenAI LLM instance
        database_tool: Database search tool
        memory: Conversation memory
        
    Returns:
        AgentExecutor for database searches
    """
    
    # Get schema information
    schema_info = database_tool.schema_info if hasattr(database_tool, 'schema_info') else "Schema not available"
    
    # Define prompt for Database agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a Database Query Expert Agent specialized in converting natural language to SQL.

Your role: Convert user questions into SQL SELECT queries and retrieve data from Azure SQL Database.

Database Schema:
{schema_info}

Rules:
- ONLY generate SELECT queries (no INSERT, UPDATE, DELETE)
- Use proper SQL syntax for Azure SQL Server
- Join tables when necessary to answer the question
- Use WHERE clauses to filter data appropriately
- Limit results to reasonable numbers (use TOP clause)
- If the question cannot be answered with available tables, explain why
- Format query results in a clear, readable way
- After showing results, provide a brief explanation of what the data means

When answering:
1. Analyze the user's question
2. Identify which tables and columns are needed
3. Construct a valid SQL SELECT query
4. Use the database_search tool with your SQL query
5. Present the results clearly
6. Explain the findings

Example:
User: "How many users are active?"
SQL: SELECT COUNT(*) as ActiveUsers FROM Users WHERE Status = 'Active'
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_functions_agent(llm, [database_tool], prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[database_tool],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor


def create_supervisor_agent(llm: AzureChatOpenAI, confluence_agent, github_agent, database_agent, memory: ConversationBufferMemory):
    """
    Create supervisor agent that coordinates between Confluence, GitHub, and Database agents
    
    Args:
        llm: Azure OpenAI LLM instance
        confluence_agent: Confluence search agent executor
        github_agent: GitHub search agent executor  
        database_agent: Database search agent executor
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
            # Detect query type based on keywords
            query_lower = query.lower()
            
            # Database query indicators
            database_keywords = [
                'count', 'how many', 'total', 'sum', 'average', 'list all',
                'show me', 'database', 'table', 'records', 'rows',
                'select', 'query', 'sql', 'customers', 'users', 'orders',
                'data from', 'in the database'
            ]
            
            is_database_query = any(keyword in query_lower for keyword in database_keywords)
            
            # If it's clearly a database query but no database agent available
            if is_database_query and not database_agent:
                return {
                    'answer': "This appears to be a database query, but the database connection is not configured.\n\n"
                             "To enable database queries:\n"
                             "1. Install ODBC Driver 18 for SQL Server\n"
                             "2. Configure database credentials in .env file:\n"
                             "   - AZURE_SQL_SERVER\n"
                             "   - AZURE_SQL_DATABASE\n"
                             "   - AZURE_SQL_USERNAME\n"
                             "   - AZURE_SQL_PASSWORD\n"
                             "3. Restart the application\n\n"
                             "For now, I can only search Confluence documentation and GitHub repositories.",
                    'confluence_used': False,
                    'github_used': False,
                    'database_used': False
                }
            
            # If it's clearly a database query, try database first
            if is_database_query and database_agent:
                logger.info("Detected database query. Querying Database agent first...")
                database_result = database_agent.invoke({"input": query})
                database_output = database_result.get('output', '')
                logger.info(f"Database agent returned {len(database_output)} characters")
                
                # Check if database query was successful
                error_phrases = ['error', 'failed', 'cannot', 'unable']
                if database_output and not any(phrase in database_output.lower() for phrase in error_phrases):
                    return {
                        'answer': database_output,
                        'confluence_used': False,
                        'github_used': False,
                        'database_used': True
                    }
            
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
            database_output = ""
            
            if is_insufficient and github_agent:
                # Query GitHub agent as fallback
                logger.info("Confluence results insufficient. Querying GitHub agent...")
                github_result = github_agent.invoke({"input": query})
                github_output = github_result.get('output', '')
                logger.info(f"GitHub agent returned {len(github_output)} characters")
                
                # Check if GitHub also insufficient
                is_github_insufficient = (
                    any(phrase in github_output.lower() for phrase in insufficient_phrases) or
                    len(github_output.strip()) < 100
                )
                
                if is_github_insufficient and database_agent:
                    # Query Database agent as final fallback
                    logger.info("GitHub results also insufficient. Querying Database agent...")
                    database_result = database_agent.invoke({"input": query})
                    database_output = database_result.get('output', '')
                    logger.info(f"Database agent returned {len(database_output)} characters")
            
            # Combine results
            outputs = []
            if confluence_output and not is_insufficient:
                outputs.append(("Confluence Documentation", confluence_output))
            if github_output:
                outputs.append(("GitHub Repositories", github_output))
            if database_output:
                outputs.append(("Database", database_output))
            
            if outputs:
                if len(outputs) == 1:
                    combined_answer = outputs[0][1]
                else:
                    combined_answer = "Based on available sources:\n\n"
                    for source_name, content in outputs:
                        combined_answer += f"**From {source_name}:**\n{content}\n\n"
            else:
                combined_answer = "I couldn't find relevant information in Confluence, GitHub repositories, or the database."
            
            return {
                'answer': combined_answer,
                'confluence_used': bool(confluence_output and not is_insufficient),
                'github_used': bool(github_output),
                'database_used': bool(database_output)
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor agent: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'confluence_used': False,
                'github_used': False
            }
    
    return supervisor

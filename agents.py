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
    Create supervisor agent that runs all agents in parallel and merges results
    
    Args:
        llm: Azure OpenAI LLM instance
        confluence_agent: Confluence search agent executor
        github_agent: GitHub search agent executor  
        database_agent: Database search agent executor
        memory: Conversation memory
        
    Returns:
        Supervisor function that orchestrates parallel execution
    """
    
    async def supervisor(query: str) -> dict:
        """
        Run all agents in parallel and merge results
        
        Args:
            query: User query
            
        Returns:
            Dict with combined answer and usage flags
        """
        import asyncio
        
        try:
            logger.info(f"Running all agents in parallel for query: {query}")
            
            # Define async wrappers for each agent
            async def query_confluence():
                """Query Confluence agent"""
                try:
                    if not confluence_agent:
                        return None
                    result = await asyncio.to_thread(confluence_agent.invoke, {"input": query})
                    output = result.get('output', '')
                    logger.info(f"Confluence returned {len(output)} characters")
                    
                    # Check if result has relevant information
                    insufficient_phrases = [
                        'no relevant information',
                        'not found',
                        'did not return',
                        'no specific information',
                        'could not find'
                    ]
                    
                    has_content = (
                        output and 
                        len(output.strip()) > 50 and
                        not any(phrase in output.lower() for phrase in insufficient_phrases)
                    )
                    
                    return output if has_content else None
                except Exception as e:
                    logger.error(f"Confluence agent error: {e}")
                    return None
            
            async def query_github():
                """Query GitHub agent"""
                try:
                    if not github_agent:
                        return None
                    result = await asyncio.to_thread(github_agent.invoke, {"input": query})
                    output = result.get('output', '')
                    logger.info(f"GitHub returned {len(output)} characters")
                    
                    # Check if result has relevant information
                    insufficient_phrases = [
                        'no relevant',
                        'not found',
                        'could not find',
                        'no repositories'
                    ]
                    
                    has_content = (
                        output and 
                        len(output.strip()) > 50 and
                        not any(phrase in output.lower() for phrase in insufficient_phrases)
                    )
                    
                    return output if has_content else None
                except Exception as e:
                    logger.error(f"GitHub agent error: {e}")
                    return None
            
            async def query_database():
                """Query Database agent"""
                try:
                    if not database_agent:
                        return None
                    result = await asyncio.to_thread(database_agent.invoke, {"input": query})
                    output = result.get('output', '')
                    logger.info(f"Database returned {len(output)} characters")
                    
                    # Check if query was successful
                    error_phrases = [
                        'error', 
                        'failed', 
                        'cannot', 
                        'unable',
                        'no tables found',
                        'not available'
                    ]
                    
                    has_content = (
                        output and 
                        len(output.strip()) > 50 and
                        not any(phrase in output.lower() for phrase in error_phrases)
                    )
                    
                    return output if has_content else None
                except Exception as e:
                    logger.error(f"Database agent error: {e}")
                    return None
            
            # Run all agents in parallel
            results = await asyncio.gather(
                query_confluence(),
                query_github(),
                query_database(),
                return_exceptions=True
            )
            
            confluence_output, github_output, database_output = results
            
            # Handle exceptions
            if isinstance(confluence_output, Exception):
                logger.error(f"Confluence exception: {confluence_output}")
                confluence_output = None
            if isinstance(github_output, Exception):
                logger.error(f"GitHub exception: {github_output}")
                github_output = None
            if isinstance(database_output, Exception):
                logger.error(f"Database exception: {database_output}")
                database_output = None
            
            # Collect valid outputs
            outputs = []
            if confluence_output:
                outputs.append(("Confluence Documentation", confluence_output))
            if github_output:
                outputs.append(("GitHub Repositories", github_output))
            if database_output:
                outputs.append(("Database", database_output))
            
            # Merge results using LLM orchestrator
            if len(outputs) == 0:
                combined_answer = "I couldn't find relevant information from any of the available sources (Confluence, GitHub, or Database)."
            elif len(outputs) == 1:
                # Single source - return as is
                combined_answer = outputs[0][1]
            else:
                # Multiple sources - use LLM to merge intelligently
                logger.info(f"Merging results from {len(outputs)} sources using LLM orchestrator")
                
                merge_prompt = f"""You are an orchestrator agent that merges information from multiple sources.

User Query: {query}

Available Information:
"""
                for source_name, content in outputs:
                    merge_prompt += f"\n--- From {source_name} ---\n{content}\n"
                
                merge_prompt += """
Instructions:
1. Synthesize the information from all sources into a coherent, comprehensive answer
2. Remove duplicate information
3. Organize the answer logically
4. If sources conflict, mention the discrepancy
5. Cite which source each piece of information came from
6. Keep it concise but complete

Provide the final merged answer:"""
                
                try:
                    orchestrator_response = await asyncio.to_thread(
                        llm.invoke,
                        merge_prompt
                    )
                    combined_answer = orchestrator_response.content
                except Exception as e:
                    logger.error(f"Orchestrator merge error: {e}")
                    # Fallback to simple concatenation
                    combined_answer = "**Combined Information from Multiple Sources:**\n\n"
                    for source_name, content in outputs:
                        combined_answer += f"**From {source_name}:**\n{content}\n\n"
            
            return {
                'answer': combined_answer,
                'confluence_used': bool(confluence_output),
                'github_used': bool(github_output),
                'database_used': bool(database_output)
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor agent: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'confluence_used': False,
                'github_used': False,
                'database_used': False
            }
    
    return supervisor

from langchain.agents import Tool
from langchain.tools import BaseTool
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ConfluenceSearchInput(BaseModel):
    """Input for Confluence search tool"""
    query: str = Field(description="The search query to find relevant Confluence documentation")


class ConfluenceSearchTool(BaseTool):
    """Tool for searching Confluence documentation"""
    name: str = "confluence_search"
    description: str = """Useful for searching internal Confluence documentation, wiki pages, and PDF attachments.
    Use this tool when the user asks about:
    - Internal documentation
    - Company processes
    - Technical specifications
    - Project documentation
    - Knowledge base articles
    Input should be a search query string."""
    args_schema: type[BaseModel] = ConfluenceSearchInput
    vector_store: object = None
    
    def _run(self, query: str) -> str:
        """Search Confluence documentation"""
        try:
            if not self.vector_store:
                return "Confluence search is not available. Vector store not initialized."
            
            # Search using vector store
            results = self.vector_store.search(query, k=5)
            
            if not results:
                return "No relevant information found in Confluence documentation."
            
            # Format results
            output = "Found the following relevant Confluence documentation:\n\n"
            for i, result in enumerate(results[:3], 1):
                output += f"{i}. **{result['title']}** ({result['type']} in {result['space']})\n"
                output += f"   Content: {result['text'][:300]}...\n"
                output += f"   URL: {result['url']}\n"
                output += f"   Relevance: {result['relevance_score']:.2%}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Confluence search: {e}")
            return f"Error searching Confluence: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


class GitHubSearchInput(BaseModel):
    """Input for GitHub search tool"""
    query: str = Field(description="The search query to find relevant GitHub repositories")


class GitHubSearchTool(BaseTool):
    """Tool for searching GitHub repositories"""
    name: str = "github_search"
    description: str = """Useful for searching GitHub repositories, README files, and code documentation.
    Use this tool when the user asks about:
    - Code examples
    - Open source projects
    - Repository information
    - Implementation details in code
    - When Confluence documentation is insufficient
    Input should be a search query string."""
    args_schema: type[BaseModel] = GitHubSearchInput
    github_searcher: object = None
    
    def _run(self, query: str) -> str:
        """Search GitHub repositories"""
        try:
            logger.info(f"GitHub Tool searching for: '{query}'")
            if not self.github_searcher:
                return "GitHub search is not available. Please configure GITHUB_TOKEN."
            
            # Search repositories
            repos = self.github_searcher.search_repositories(query, max_results=3)
            logger.info(f"GitHub Tool found {len(repos)} repositories")
            
            if not repos:
                logger.info("GitHub Tool: No repositories found")
                return "No relevant GitHub repositories found in your accessible repos."
            
            # Format results
            output = "Found the following relevant GitHub repositories:\n\n"
            for i, repo in enumerate(repos, 1):
                logger.info(f"GitHub Repo {i}: {repo['name']} (stars: {repo['stars']}, score: {repo.get('score', 0)})")
                logger.info(f"  Description: {repo['description']}")
                logger.info(f"  README length: {len(repo['readme'])} chars")
                output += f"{i}. **{repo['name']}** ({repo['stars']} ⭐)\n"
                output += f"   Description: {repo['description']}\n"
                output += f"   Language: {repo['language']}\n"
                output += f"   Topics: {', '.join(repo['topics'][:5])}\n"
                output += f"   README Preview: {repo['readme'][:400]}...\n"
                output += f"   URL: {repo['url']}\n"
                output += f"   Private: {repo.get('private', False)}\n\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error in GitHub search: {e}")
            return f"Error searching GitHub: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)


def create_confluence_tool(vector_store) -> ConfluenceSearchTool:
    """Create Confluence search tool with vector store"""
    return ConfluenceSearchTool(vector_store=vector_store)


def create_github_tool(github_searcher) -> GitHubSearchTool:
    """Create GitHub search tool with searcher"""
    return GitHubSearchTool(github_searcher=github_searcher)

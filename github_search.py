from github import Github
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubSearcher:
    """Search GitHub repositories for relevant information"""
    
    def __init__(self, github_token: Optional[str] = None, organization: Optional[str] = None):
        """
        Initialize GitHub searcher
        Args:
            github_token: GitHub personal access token (required for private repos)
            organization: Optional organization name to limit search scope
        """
        if not github_token:
            raise ValueError("GitHub token is required to search private/accessible repositories")
        
        self.github = Github(github_token)
        self.organization = organization
        self.user = self.github.get_user()
        logger.info(f"Authenticated as: {self.user.login}")
    
    def get_accessible_repos(self) -> List[Dict]:
        """
        Get all repositories accessible to the authenticated user
        Returns:
            List of accessible repository names
        """
        try:
            accessible_repos = []
            
            # Get user's own repos
            for repo in self.user.get_repos():
                accessible_repos.append(repo.full_name)
            
            logger.info(f"Found {len(accessible_repos)} accessible repositories")
            return accessible_repos
            
        except Exception as e:
            logger.error(f"Error getting accessible repos: {e}")
            return []
    
    def search_repositories(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search GitHub repositories based on query (searches in description AND README content)
        First uses GitHub search API, then enhances with README content analysis
        Args:
            query: Search query
            max_results: Maximum number of repositories to return
        Returns:
            List of repository information dicts
        """
        try:
            results = []
            query_lower = query.lower()
            query_words = set(word.lower() for word in query.split() if len(word) > 3)
            
            # Build search query to limit to user's accessible repos
            if self.organization:
                search_query = f"{query} org:{self.organization}"
            else:
                # Search in user's repos
                search_query = f"{query} user:{self.user.login}"
            
            logger.info(f"Searching GitHub with query: {search_query}")
            
            # Use GitHub's search API first (faster, more targeted)
            repos = self.github.search_repositories(
                query=search_query, 
                sort='stars',  # Sort by stars for quality
                order='desc'
            )
            
            # Process top results and check README
            scored_repos = []
            
            for i, repo in enumerate(repos):
                if i >= max_results * 2:  # Check 2x max_results to have buffer
                    break
                
                try:
                    score = 10  # Base score for being in search results
                    
                    # Get repository metadata
                    repo_name = repo.full_name.lower()
                    repo_description = (repo.description or "").lower()
                    repo_topics = [t.lower() for t in repo.get_topics()]
                    
                    # Boost score for exact matches in name
                    if query_lower in repo_name:
                        score += 20
                    
                    # Check each query word in name
                    for word in query_words:
                        if word in repo_name:
                            score += 10
                    
                    # Check description
                    if query_lower in repo_description:
                        score += 10
                    
                    for word in query_words:
                        if word in repo_description:
                            score += 5
                    
                    # Check topics
                    for topic in repo_topics:
                        if query_lower in topic:
                            score += 8
                        for word in query_words:
                            if word in topic:
                                score += 4
                    
                    # Get README content
                    readme_content = ""
                    readme_raw = ""
                    try:
                        readme = repo.get_readme()
                        readme_raw = readme.decoded_content.decode('utf-8')
                        readme_content = readme_raw.lower()
                        logger.info(f"README for {repo.full_name}: {readme_raw[:200]}...")
                    except Exception as e:
                        logger.debug(f"No README for {repo.full_name}: {e}")
                        readme_raw = "No README available"
                        readme_content = ""
                    
                    # Check README content (important for detailed matching)
                    if readme_content:
                        # Full query match in README
                        if query_lower in readme_content:
                            score += 15
                            logger.info(f"✓ Found '{query_lower}' in README of {repo.full_name}")
                        
                        # Individual words in README
                        for word in query_words:
                            if word in readme_content:
                                score += 3
                                logger.info(f"✓ Found word '{word}' in README of {repo.full_name}")
                    
                    logger.info(f"Repository: {repo.full_name}")
                    logger.info(f"  Name match: {query_lower in repo_name}")
                    logger.info(f"  Description: {repo.description}")
                    logger.info(f"  Topics: {repo_topics}")
                    logger.info(f"  README length: {len(readme_raw)} chars")
                    logger.info(f"  Final score: {score}")
                    
                    repo_info = {
                        'name': repo.full_name,
                        'url': repo.html_url,
                        'description': repo.description or "No description",
                        'readme': readme_raw[:2000],  # First 2000 chars
                        'stars': repo.stargazers_count,
                        'language': repo.language,
                        'topics': repo.get_topics(),
                        'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                        'private': repo.private,
                        'score': score
                    }
                    scored_repos.append(repo_info)
                    logger.info(f"Found: {repo.full_name} (score: {score}, stars: {repo.stargazers_count})")
                    
                except Exception as e:
                    logger.error(f"Error processing repo {repo.full_name}: {e}")
                    continue
            
            # Sort by score (highest first) and return top results
            scored_repos.sort(key=lambda x: x['score'], reverse=True)
            results = scored_repos[:max_results]
            
            logger.info(f"Returning {len(results)} repositories")
            for r in results:
                logger.info(f"  - {r['name']} (score: {r['score']})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            return []
    
    def get_repository_info(self, repo_full_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific repository
        Args:
            repo_full_name: Repository name in format 'owner/repo'
        Returns:
            Dictionary with repository information
        """
        try:
            repo = self.github.get_repo(repo_full_name)
            
            # Get README
            readme_content = ""
            try:
                readme = repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
            except:
                readme_content = "No README available"
            
            repo_info = {
                'name': repo.full_name,
                'url': repo.html_url,
                'description': repo.description or "No description",
                'readme': readme_content,
                'stars': repo.stargazers_count,
                'language': repo.language,
                'topics': repo.get_topics(),
                'updated_at': repo.updated_at.isoformat() if repo.updated_at else None
            }
            
            return repo_info
            
        except Exception as e:
            logger.error(f"Error getting repo info for {repo_full_name}: {e}")
            return None
    
    def search_code(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search code files in GitHub
        Args:
            query: Code search query
            max_results: Maximum results to return
        Returns:
            List of code search results
        """
        try:
            results = []
            code_results = self.github.search_code(query=query)
            
            for i, code in enumerate(code_results):
                if i >= max_results:
                    break
                
                try:
                    result = {
                        'name': code.name,
                        'path': code.path,
                        'repository': code.repository.full_name,
                        'url': code.html_url,
                        'repo_url': code.repository.html_url
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing code result: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching code: {e}")
            return []
    
    def is_relevant(self, repo_info: Dict, query: str) -> bool:
        """
        Check if repository is relevant to the query
        Args:
            repo_info: Repository information dict
            query: User query
        Returns:
            Boolean indicating relevance
        """
        query_lower = query.lower()
        
        # Check in name, description, topics, and README
        check_fields = [
            repo_info.get('name', '').lower(),
            repo_info.get('description', '').lower(),
            ' '.join(repo_info.get('topics', [])).lower(),
            repo_info.get('readme', '')[:1000].lower()
        ]
        
        # Simple keyword matching
        for field in check_fields:
            if any(word in field for word in query_lower.split() if len(word) > 3):
                return True
        
        return False

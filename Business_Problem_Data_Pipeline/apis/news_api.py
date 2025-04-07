"""NewsAPI client for collecting industry problems."""

import logging
from datetime import datetime, timedelta
from newsapi import NewsApiClient

from config import NEWS_API_KEY, NEWS_KEYWORDS, NEWS_API_RATE_LIMIT
from apis.base import BaseAPIClient
from db.models import BusinessProblem

logger = logging.getLogger(__name__)

class NewsAPIClient:
    """Client for the NewsAPI."""
    
    def __init__(self):
        """Initialize the NewsAPI client."""
        self.client = NewsApiClient(api_key=NEWS_API_KEY)
        logger.info("NewsAPI client initialized")
    
    def fetch_problems(self, keywords=None, days=7, limit=100):
        """
        Fetch news articles related to business problems.
        
        Args:
            keywords (list): Keywords to search for
            days (int): Number of days to look back
            limit (int): Maximum number of articles to fetch
            
        Returns:
            list: List of BusinessProblem objects
        """
        if keywords is None:
            keywords = NEWS_KEYWORDS
        
        problems = []
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        for keyword in keywords:
            try:
                logger.info(f"Fetching news articles for keyword: {keyword}")
                
                # Get articles
                response = self.client.get_everything(
                    q=keyword,
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=min(100, limit)
                )
                
                if not response or 'articles' not in response:
                    logger.warning(f"No articles returned for keyword: {keyword}")
                    continue
                
                # Process articles
                for article in response['articles'][:limit]:
                    if self._is_problem_article(article):
                        problem = self._article_to_problem(article, keyword)
                        problems.append(problem)
                
                logger.info(f"Fetched {len(problems)} problem articles for keyword: {keyword}")
                
            except Exception as e:
                logger.error(f"Error fetching news for keyword {keyword}: {e}")
        
        return problems
    
    def _is_problem_article(self, article):
        """
        Check if an article is likely about a business problem.
        
        Args:
            article (dict): Article data
            
        Returns:
            bool: True if the article is likely about a business problem
        """
        # Check title and description for problem indicators
        problem_indicators = [
            "challenge", "issue", "problem", "struggle", "difficulty",
            "obstacle", "hurdle", "barrier", "bottleneck", "gap",
            "weakness", "vulnerability", "threat", "risk", "concern"
        ]
        
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        for indicator in problem_indicators:
            if indicator in title or indicator in description:
                return True
        
        return False
    
    def _article_to_problem(self, article, keyword):
        """
        Convert a news article to a BusinessProblem.
        
        Args:
            article (dict): Article data
            keyword (str): Keyword used to find the article
            
        Returns:
            BusinessProblem: Business problem object
        """
        # Build text from title and description
        text = f"{article.get('title', '')}\n\n{article.get('description', '')}"
        if article.get('content'):
            text += f"\n\n{article.get('content')}"
        
        # Extract source name
        source_name = article.get('source', {}).get('name', 'unknown')
        
        # Create metadata
        metadata = {
            'keyword': keyword,
            'source_name': source_name,
            'author': article.get('author'),
            'published_at': article.get('publishedAt')
        }
        
        # Get tags (categories)
        tags = [keyword, source_name]
        
        # Create business problem
        return BusinessProblem(
            text=text,
            source="news_api",
            source_id=article.get('url'),  # Use URL as unique ID
            title=article.get('title'),
            url=article.get('url'),
            author=article.get('author'),
            created_at=datetime.fromisoformat(article.get('publishedAt').replace('Z', '+00:00'))
            if article.get('publishedAt') else datetime.now(),
            tags=tags,
            metadata=metadata
        )
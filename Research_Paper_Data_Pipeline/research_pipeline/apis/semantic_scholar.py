"""
Semantic Scholar API client for fetching research papers.
"""

import requests
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from research_pipeline.apis.base import BaseAPIClient
from research_pipeline.models.research_paper import ResearchPaper, Author

logger = logging.getLogger(__name__)

class SemanticScholarAPIClient(BaseAPIClient):
    """API client for Semantic Scholar."""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Semantic Scholar")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key
        self.headers = {
            "User-Agent": "Prospectis/1.0",
        }
        
        if api_key:
            self.headers["x-api-key"] = api_key
    
    def fetch_recent_papers(self, days_back: int = 1, limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch recent papers from Semantic Scholar API.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to retrieve
            
        Returns:
            List of ResearchPaper objects
        """
        logger.info(f"Fetching papers from Semantic Scholar published in the last {days_back} days (limit: {limit})")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Using the paper search endpoint
        search_query = "machine learning OR artificial intelligence OR computer science"
        search_url = f"{self.base_url}/paper/search"
        
        params = {
            "query": search_query,
            "limit": limit,
            "fields": "title,abstract,authors,year,venue,publicationDate,url,externalIds"
        }
        
        try:
            response = self._make_request(search_url, params)
            
            # Add robust error handling
            if not response or not isinstance(response, dict) or 'data' not in response:
                logger.warning("Unexpected Semantic Scholar API response format")
                return []
            
            papers = []
            for paper in response.get('data', []):
                if not paper:
                    continue
                
                try:
                    # Extract author information - convert to Author objects
                    author_objects = []
                    if paper.get('authors'):
                        for author_data in paper['authors']:
                            if isinstance(author_data, dict) and author_data.get('name'):
                                author_name = author_data.get('name')
                                author_objects.append(Author(name=author_name))
                            elif isinstance(author_data, str):
                                author_objects.append(Author(name=author_data))
                    
                    # Generate a paper_id if not present
                    paper_id = paper.get('paperId')
                    if not paper_id:
                        paper_id = f"ss-{uuid.uuid4()}"
                    
                    # Format the published date correctly
                    published_date = None
                    date_str = paper.get('publicationDate')
                    if date_str:
                        try:
                            # Try to parse the date - add time component if missing
                            if 'T' not in date_str and ' ' not in date_str:
                                date_str = f"{date_str}T00:00:00"
                            published_date = datetime.fromisoformat(date_str)
                        except ValueError:
                            # If parsing fails, use the current date
                            published_date = datetime.now()
                    
                    # Extract additional fields
                    title = paper.get('title', 'Unknown Title')
                    abstract = paper.get('abstract', '')
                    year = paper.get('year')
                    url = paper.get('url')
                    
                    # Extract external IDs
                    external_ids = paper.get('externalIds', {})
                    doi = external_ids.get('DOI')
                    
                    # Handle venue data
                    categories = []
                    venue = None
                    venue_data = paper.get('venue')
                    if isinstance(venue_data, dict) and venue_data.get('name'):
                        venue = venue_data.get('name')
                    elif isinstance(venue_data, str):
                        venue = venue_data
                    
                    if venue:
                        categories.append(venue)
                    
                    # Create the ResearchPaper object
                    paper_obj = ResearchPaper(
                        paper_id=paper_id,
                        source="semantic_scholar",
                        source_id=paper_id,
                        title=title,
                        abstract=abstract,
                        authors=author_objects,
                        published_date=published_date,
                        url=url,
                        doi=doi,
                        categories=categories,
                        journal=venue,
                        metadata={"original_response": paper}
                    )
                    papers.append(paper_obj)
                except Exception as e:
                    logger.error(f"Error processing paper: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            logger.info(f"Successfully fetched {len(papers)} papers from Semantic Scholar")
            return papers
        except Exception as e:
            logger.error(f"Error fetching papers from Semantic Scholar: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the Semantic Scholar API with rate limiting.
        
        Args:
            url: API endpoint URL
            params: URL parameters
            
        Returns:
            API response as dictionary
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, headers=self.headers)
                
                if response.status_code == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                    
                if response.status_code == 403:
                    logger.error("Access forbidden. Check API key and permissions.")
                    return {}
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {str(e)}")
                retry_count += 1
                time.sleep(2 ** retry_count)  # Exponential backoff
                
        logger.error(f"Failed to make request after {max_retries} retries")
        return {}

# Alias for compatibility with main.py
class SemanticScholarClient(SemanticScholarAPIClient):
    """Alias for SemanticScholarAPIClient for compatibility."""
    pass

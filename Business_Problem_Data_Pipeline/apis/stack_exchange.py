"""Stack Exchange API client for collecting questions."""

import logging
import requests
from datetime import datetime, timedelta

from Business_Problem_Data_Pipeline.config import (
    STACK_EXCHANGE_KEY, STACK_EXCHANGE_SITES, STACK_EXCHANGE_RATE_LIMIT
)
from Business_Problem_Data_Pipeline.apis.base import BaseAPIClient
from Business_Problem_Data_Pipeline.db.models import BusinessProblem

logger = logging.getLogger(__name__)

class StackExchangeClient(BaseAPIClient):
    """Client for the Stack Exchange API."""
    
    def __init__(self):
        """Initialize the Stack Exchange client."""
        super().__init__(
            base_url="https://api.stackexchange.com/2.3",
            rate_limit=STACK_EXCHANGE_RATE_LIMIT,
            source_name="StackExchange"
        )
        self.api_key = STACK_EXCHANGE_KEY
        logger.info("Stack Exchange client initialized")
    
    def _get_headers(self):
        """Get headers for API requests."""
        headers = super()._get_headers()
        headers.update({
            "Accept": "application/json"
        })
        return headers
    
    def fetch_problems(self, sites=None, days=7, limit=100):
        """
        Fetch questions from Stack Exchange sites.
        
        Args:
            sites (list, optional): List of Stack Exchange sites to fetch from
            days (int, optional): Number of days to look back
            limit (int, optional): Maximum number of questions to fetch per site
            
        Returns:
            list: List of BusinessProblem objects
        """
        if sites is None:
            sites = STACK_EXCHANGE_SITES
        
        problems = []
        from_date = int((datetime.now() - timedelta(days=days)).timestamp())
        
        for site in sites:
            try:
                logger.info(f"Fetching questions from {site}")
                
                # Prepare parameters
                params = {
                    "site": site,
                    "key": self.api_key,
                    "order": "desc",
                    "sort": "activity",
                    "filter": "withbody",  # Include the question body
                    "fromdate": from_date,
                    "pagesize": min(100, limit)  # Max allowed is 100
                }
                
                # Get questions
                page = 1
                total_fetched = 0
                
                while total_fetched < limit:
                    params["page"] = page
                    response = self.get("questions", params=params)
                    
                    if not response or "items" not in response:
                        logger.warning(f"No items returned from {site} API")
                        break
                    
                    # Process questions
                    for question in response["items"]:
                        problem = self._question_to_problem(question, site)
                        problems.append(problem)
                        total_fetched += 1
                    
                    # Check if there are more pages
                    if not response.get("has_more", False) or total_fetched >= limit:
                        break
                    
                    page += 1
                
                logger.info(f"Fetched {total_fetched} questions from {site}")
                
            except Exception as e:
                logger.error(f"Error fetching from {site}: {e}")
        
        return problems
    
    def _question_to_problem(self, question, site):
        """
        Convert a Stack Exchange question to a BusinessProblem.
        
        Args:
            question (dict): Question data from the API
            site (str): Stack Exchange site name
            
        Returns:
            BusinessProblem: Business problem object
        """
        # Create metadata
        metadata = {
            "site": site,
            "score": question.get("score", 0),
            "view_count": question.get("view_count", 0),
            "answer_count": question.get("answer_count", 0),
            "is_answered": question.get("is_answered", False)
        }
        
        # Get tags
        tags = question.get("tags", [])
        tags.append(site)
        
        # Create business problem
        return BusinessProblem(
            text=f"{question.get('title', '')}\n\n{question.get('body', '')}",
            source="stack_exchange",
            source_id=str(question.get("question_id")),
            title=question.get("title"),
            url=question.get("link"),
            author=question.get("owner", {}).get("display_name") if "owner" in question else None,
            created_at=datetime.fromtimestamp(question.get("creation_date", 0)),
            tags=tags,
            metadata=metadata
        )
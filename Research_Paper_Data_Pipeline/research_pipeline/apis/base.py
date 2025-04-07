"""
Base API client class for research paper sources.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Generator
import backoff
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout

from research_pipeline.models.research_paper import ResearchPaper


class BaseAPIClient(ABC):
    """Base class for all API clients."""
    
    def __init__(self, name: str):
        """
        Initialize the base API client.
        
        Args:
            name (str): Name of the API client (used for logging)
        """
        self.name = name
        self.session = requests.Session()
        logger.info(f"Initialized {self.name} API client")
    
    @abstractmethod
    def fetch_recent_papers(self, 
                           days_back: int = 3, 
                           limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers published in the last n days.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to fetch
            
        Returns:
            List of ResearchPaper objects
        """
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, Timeout))
    )
    def _make_request(self, 
                     url: str, 
                     method: str = "GET", 
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     json_data: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        Make an HTTP request with retry logic.
        
        Args:
            url: Request URL
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            headers: HTTP headers
            data: Form data
            json_data: JSON data
            
        Returns:
            Response object
        """
        method = method.upper()
        headers = headers or {}
        
        logger.debug(f"{self.name}: Making {method} request to {url}")
        
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            data=data,
            json=json_data,
            timeout=30  # 30 second timeout
        )
        
        # Raise for HTTP errors (400+)
        response.raise_for_status()
        
        # Rate limiting - check headers for rate limit info
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining and int(remaining) < 10:
            logger.warning(f"{self.name}: Rate limit almost reached. Remaining: {remaining}")
            # Sleep if rate limit is getting low
            time.sleep(2)
        
        return response
    
    def get_date_n_days_ago(self, n: int) -> datetime:
        """
        Get a datetime object for n days ago.
        
        Args:
            n: Number of days to go back
            
        Returns:
            datetime object for n days ago
        """
        return datetime.utcnow() - timedelta(days=n)
    
    def close(self):
        """Close the API client session."""
        self.session.close()
        logger.debug(f"Closed {self.name} API client session")
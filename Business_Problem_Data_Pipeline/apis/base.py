"""Base API client with rate limiting and error handling."""

import logging
import requests
from Business_Problem_Data_Pipeline.utils.rate_limiter import RateLimiter, ExponentialBackoff

logger = logging.getLogger(__name__)

class BaseAPIClient:
    """Base class for API clients."""
    
    def __init__(self, base_url, rate_limit, source_name):
        """
        Initialize the base API client.
        
        Args:
            base_url (str): Base URL for API requests
            rate_limit (int): Requests per minute rate limit
            source_name (str): Name of the data source
        """
        self.base_url = base_url
        self.rate_limiter = RateLimiter(rate_limit)
        self.backoff = ExponentialBackoff()
        self.source_name = source_name
        self.session = requests.Session()
    
    def _get_headers(self):
        """
        Get headers for API requests.
        
        Returns:
            dict: Headers dictionary
        """
        return {
            "User-Agent": f"Prospectis/1.0 Data Collection Bot"
        }
    
    def _request(self, method, endpoint, params=None, data=None, headers=None, auth=None):
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            params (dict, optional): Query parameters
            data (dict, optional): Request body
            headers (dict, optional): HTTP headers
            auth (tuple or object, optional): Authentication credentials
            
        Returns:
            dict or None: Response data or None if request failed
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Combine headers
        request_headers = self._get_headers()
        if headers:
            request_headers.update(headers)
        
        # Apply rate limiting
        self.rate_limiter.wait()
        
        # Reset backoff
        self.backoff.reset()
        
        while True:
            try:
                logger.debug(f"Making {method} request to {url}")
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=data,
                    headers=request_headers,
                    auth=auth,
                    timeout=30
                )
                
                # Check if rate limited
                if response.status_code == 429:
                    logger.warning(f"Rate limited by {self.source_name} API")
                    if not self.backoff.wait():
                        return None
                    continue
                
                # Check for other errors
                if response.status_code >= 400:
                    logger.error(f"{self.source_name} API error: {response.status_code} - {response.text}")
                    if response.status_code >= 500:  # Server error, retry
                        if not self.backoff.wait():
                            return None
                        continue
                    return None
                
                # Parse JSON response
                return response.json()
                
            except (requests.RequestException, ValueError) as e:
                logger.error(f"{self.source_name} API request failed: {e}")
                if not self.backoff.wait():
                    return None
    
    def get(self, endpoint, params=None, headers=None, auth=None):
        """Make a GET request."""
        return self._request("GET", endpoint, params=params, headers=headers, auth=auth)
    
    def post(self, endpoint, data=None, params=None, headers=None, auth=None):
        """Make a POST request."""
        return self._request("POST", endpoint, params=params, data=data, headers=headers, auth=auth)
    
    def close(self):
        """Close the session."""
        self.session.close()
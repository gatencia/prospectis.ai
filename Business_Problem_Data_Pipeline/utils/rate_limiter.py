"""Rate limiting utilities for API requests."""

import time
import logging
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_period, period=60):
        """
        Initialize a rate limiter.
        
        Args:
            requests_per_period (int): Maximum number of requests allowed in the period
            period (int, optional): Period in seconds. Defaults to 60.
        """
        self.requests_per_period = requests_per_period
        self.period = period
        self.request_timestamps = deque()
    
    def wait(self):
        """
        Wait if necessary to stay within rate limits.
        """
        now = datetime.now()
        
        # Remove timestamps older than the period
        while self.request_timestamps and self.request_timestamps[0] < now - timedelta(seconds=self.period):
            self.request_timestamps.popleft()
        
        # If we've reached the limit, wait until we can make another request
        if len(self.request_timestamps) >= self.requests_per_period:
            oldest = self.request_timestamps[0]
            sleep_time = (oldest + timedelta(seconds=self.period) - now).total_seconds()
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Record this request
        self.request_timestamps.append(datetime.now())

class ExponentialBackoff:
    """Exponential backoff for retrying failed requests."""
    
    def __init__(self, initial_delay=1, max_delay=60, max_retries=5):
        """
        Initialize exponential backoff.
        
        Args:
            initial_delay (int, optional): Initial delay in seconds. Defaults to 1.
            max_delay (int, optional): Maximum delay in seconds. Defaults to 60.
            max_retries (int, optional): Maximum number of retries. Defaults to 5.
        """
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.retry_count = 0
    
    def reset(self):
        """Reset retry count."""
        self.retry_count = 0
    
    def wait(self):
        """
        Wait with exponential backoff.
        
        Returns:
            bool: False if max retries reached, True otherwise
        """
        if self.retry_count >= self.max_retries:
            logger.warning(f"Max retries ({self.max_retries}) reached")
            return False
        
        delay = min(self.initial_delay * (2 ** self.retry_count), self.max_delay)
        logger.debug(f"Backing off for {delay} seconds (retry {self.retry_count + 1}/{self.max_retries})")
        time.sleep(delay)
        self.retry_count += 1
        return True
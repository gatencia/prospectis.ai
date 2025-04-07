"""Reddit API client for collecting problem posts."""

import logging
import praw
from datetime import datetime, timedelta

from Business_Problem_Data_Pipeline.config import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS, REDDIT_RATE_LIMIT
)
from Business_Problem_Data_Pipeline.db.models import BusinessProblem

logger = logging.getLogger(__name__)

class RedditClient:
    """Client for the Reddit API."""
    
    def __init__(self):
        """Initialize the Reddit client."""
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        logger.info("Reddit client initialized")
    
    def fetch_problems(self, subreddits=None, time_filter="day", limit=100):
        """
        Fetch problem posts from Reddit.
        
        Args:
            subreddits (list, optional): List of subreddits to fetch from
            time_filter (str, optional): Time filter (hour, day, week, month, year, all)
            limit (int, optional): Maximum number of posts to fetch per subreddit
            
        Returns:
            list: List of BusinessProblem objects
        """
        if subreddits is None:
            subreddits = REDDIT_SUBREDDITS
        
        problems = []
        
        for subreddit_name in subreddits:
            try:
                logger.info(f"Fetching posts from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get posts from the subreddit
                for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    # Filter for posts that likely contain problems
                    if self._is_problem_post(submission):
                        problem = self._submission_to_problem(submission, subreddit_name)
                        problems.append(problem)
                        logger.debug(f"Found problem: {problem.title}")
                
                logger.info(f"Fetched {len(problems)} problem posts from r/{subreddit_name}")
                
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")
        
        return problems
    
    def _is_problem_post(self, submission):
        """
        Check if a submission is likely a problem post.
        
        Args:
            submission: Reddit submission object
            
        Returns:
            bool: True if the submission is likely a problem post
        """
        # Check title for problem indicators
        title_lower = submission.title.lower()
        problem_indicators = [
            "help", "issue", "problem", "error", "bug", "how to", "how do i",
            "question", "trouble", "struggling", "stuck", "can't", "doesn't work",
            "broken", "failing", "failure", "need advice", "advice needed"
        ]
        
        for indicator in problem_indicators:
            if indicator in title_lower:
                return True
        
        # Check if it's a question (ends with ?)
        if submission.title.strip().endswith("?"):
            return True
        
        # Check selftext if available
        if hasattr(submission, "selftext") and submission.selftext:
            selftext_lower = submission.selftext.lower()
            for indicator in problem_indicators:
                if indicator in selftext_lower:
                    return True
        
        return False
    
    def _submission_to_problem(self, submission, subreddit_name):
        """
        Convert a Reddit submission to a BusinessProblem.
        
        Args:
            submission: Reddit submission object
            subreddit_name (str): Name of the subreddit
            
        Returns:
            BusinessProblem: Business problem object
        """
        # Combine title and selftext for the problem text
        if hasattr(submission, "selftext") and submission.selftext:
            text = f"{submission.title}\n\n{submission.selftext}"
        else:
            text = submission.title
        
        # Create metadata
        metadata = {
            "subreddit": subreddit_name,
            "score": submission.score,
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "is_self": submission.is_self
        }
        
        # Get tags from submission flair
        tags = []
        if hasattr(submission, "link_flair_text") and submission.link_flair_text:
            tags.append(submission.link_flair_text)
        tags.append(subreddit_name)
        
        # Create business problem
        return BusinessProblem(
            text=text,
            source="reddit",
            source_id=submission.id,
            title=submission.title,
            url=f"https://www.reddit.com{submission.permalink}",
            author=submission.author.name if submission.author else "[deleted]",
            created_at=datetime.fromtimestamp(submission.created_utc),
            tags=tags,
            metadata=metadata
        )
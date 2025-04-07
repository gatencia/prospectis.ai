"""
arXiv API client for fetching research papers.
"""

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator
import arxiv
from loguru import logger

from research_pipeline.apis.base import BaseAPIClient
from research_pipeline.models.research_paper import ResearchPaper, Author


class ArxivClient(BaseAPIClient):
    """Client for interacting with the arXiv API."""
    
    # CS categories in arXiv
    CS_CATEGORIES = [
        "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", 
        "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", 
        "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", 
        "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", 
        "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", 
        "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
    ]
    
    def __init__(self):
        """Initialize the arXiv API client."""
        super().__init__(name="arXiv")
    
    def fetch_recent_papers(self, days_back: int = 3, limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers published in the last n days from arXiv.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to fetch
            
        Returns:
            List of ResearchPaper objects
        """
        logger.info(f"Fetching papers from arXiv published in the last {days_back} days (limit: {limit})")
        
        # Calculate date range
        date_since = self.get_date_n_days_ago(days_back)
        
        # Prepare the search query
        # Search for CS papers submitted/updated since the date
        date_str = date_since.strftime("%Y%m%d%H%M%S")
        query = f"cat:{'|'.join(self.CS_CATEGORIES)} AND submittedDate:[{date_str} TO now]"
        
        logger.debug(f"arXiv query: {query}")
        
        # Set up the arXiv client
        client = arxiv.Client(
            page_size=100,  # Max papers per request
            delay_seconds=3,  # Be nice to the API
            num_retries=3    # Retry on failures
        )
        
        # Create the search query
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        try:
            # Execute the search and convert results to our model
            for result in client.results(search):
                paper = self._convert_to_model(result)
                papers.append(paper)
                
                # Respect the limit
                if len(papers) >= limit:
                    break
                    
            logger.info(f"Successfully fetched {len(papers)} papers from arXiv")
            
        except Exception as e:
            logger.error(f"Error fetching papers from arXiv: {str(e)}")
            # Return what we have so far
        
        return papers
    
    def _convert_to_model(self, arxiv_result) -> ResearchPaper:
        """
        Convert an arXiv search result to our ResearchPaper model.
        
        Args:
            arxiv_result: An arXiv result object
            
        Returns:
            ResearchPaper object
        """
        # Extract the arXiv ID
        arxiv_id = arxiv_result.get_short_id()
        
        # Extract categories
        categories = [cat for cat in arxiv_result.categories]
        
        # Extract authors
        authors = [
            Author(
                name=author.name,
                affiliation=None  # arXiv doesn't provide affiliation info through the API
            )
            for author in arxiv_result.authors
        ]
        
        # Convert to our model
        return ResearchPaper(
            paper_id=f"arxiv_{arxiv_id}",
            source="arxiv",
            source_id=arxiv_id,
            title=arxiv_result.title,
            abstract=arxiv_result.summary,
            authors=authors,
            published_date=arxiv_result.published,
            url=arxiv_result.entry_id,
            doi=None,  # arXiv entries might not have a DOI
            categories=categories,
            pdf_url=arxiv_result.pdf_url,
            journal=None,  # arXiv is a preprint server
            metadata={
                "comment": arxiv_result.comment,
                "journal_ref": arxiv_result.journal_ref,
                "primary_category": arxiv_result.primary_category,
                "updated": arxiv_result.updated.isoformat() if arxiv_result.updated else None
            }
        )
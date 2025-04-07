"""
IEEE Xplore API client for fetching research papers.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
import time

from research_pipeline.apis.base import BaseAPIClient
from research_pipeline.models.research_paper import ResearchPaper, Author


class IEEEClient(BaseAPIClient):
    """Client for interacting with the IEEE Xplore API."""
    
    # Base URL for the IEEE API
    BASE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the IEEE Xplore API client.
        
        Args:
            api_key: IEEE Xplore API key (if None, will look for IEEE_API_KEY in env)
        """
        super().__init__(name="IEEE")
        
        self.api_key = api_key or os.getenv("IEEE_API_KEY")
        if not self.api_key:
            logger.warning("No IEEE API key provided. API calls will fail.")
    
    def fetch_recent_papers(self, days_back: int = 3, limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers published in the last n days from IEEE Xplore.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to fetch
            
        Returns:
            List of ResearchPaper objects
        """
        if not self.api_key:
            logger.error("Cannot fetch papers from IEEE: No API key provided")
            return []
            
        logger.info(f"Fetching papers from IEEE published in the last {days_back} days (limit: {limit})")
        
        # Calculate date range
        date_since = self.get_date_n_days_ago(days_back)
        date_str = date_since.strftime("%Y%m%d")
        
        # Prepare query parameters - filter for computer science and recent papers
        params = {
            "apikey": self.api_key,
            "format": "json",
            "max_records": min(100, limit),  # IEEE API max is 200, but let's be conservative
            "start_record": 1,
            "sort_order": "desc",
            "sort_field": "publication_date",
            "start_date": date_str,
            "end_date": datetime.now().strftime("%Y%m%d"),
            "content_type": "Conferences,Journals,Early Access",
            # Filter for CS related papers
            "query_text": "((computer science) OR (artificial intelligence) OR (machine learning) OR (software) OR (computational))"
        }
        
        logger.debug(f"IEEE query params: {params}")
        
        # Fetch papers with pagination
        papers = []
        start_record = 1
        max_records = min(100, limit)
        
        while len(papers) < limit:
            current_params = {**params, "start_record": start_record, "max_records": max_records}
            
            try:
                response = self._make_request(
                    url=self.BASE_URL,
                    params=current_params
                )
                
                data = response.json()
                results = data.get("articles", [])
                total_records = data.get("total_records", 0)
                
                if not results:
                    logger.debug("No more results from IEEE")
                    break
                
                # Process results
                for item in results:
                    paper = self._convert_to_model(item)
                    papers.append(paper)
                    
                    # Check if we've reached the limit
                    if len(papers) >= limit:
                        break
                
                # Check if we've reached the end of results
                if start_record + len(results) > total_records:
                    break
                    
                # Update for next page
                start_record += len(results)
                
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching papers from IEEE: {str(e)}")
                break
        
        logger.info(f"Successfully fetched {len(papers)} papers from IEEE")
        return papers
    
    def _convert_to_model(self, ieee_item: Dict[str, Any]) -> ResearchPaper:
        """
        Convert an IEEE item to our ResearchPaper model.
        
        Args:
            ieee_item: An IEEE item object
            
        Returns:
            ResearchPaper object
        """
        # Extract the article number as ID
        article_id = str(ieee_item.get("article_number", ""))
        
        # Extract title and abstract
        title = ieee_item.get("title", "")
        abstract = ieee_item.get("abstract", "")
        
        # Extract DOI
        doi = ieee_item.get("doi", "")
        
        # Extract authors
        authors = []
        for author_data in ieee_item.get("authors", {}).get("authors", []):
            name = author_data.get("full_name", "")
            affiliation = None
            if "affiliation" in author_data:
                affiliation = author_data["affiliation"]
                
            authors.append(Author(
                name=name,
                affiliation=affiliation
            ))
        
        # Extract publication date
        published_date = None
        pub_date_str = ieee_item.get("publication_date")
        if pub_date_str:
            try:
                # IEEE date format can vary, try different formats
                for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
                    try:
                        published_date = datetime.strptime(pub_date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Could not parse IEEE publication date {pub_date_str}: {str(e)}")
        
        # Extract URL
        url = ieee_item.get("html_url")
        
        # Extract PDF URL
        pdf_url = ieee_item.get("pdf_url")
        
        # Extract keywords/categories
        categories = []
        for terms in [
            ieee_item.get("index_terms", {}).get("ieee_terms", {}).get("terms", []),
            ieee_item.get("index_terms", {}).get("author_terms", {}).get("terms", []),
            ieee_item.get("keywords", [])
        ]:
            if terms:
                categories.extend(terms)
        
        # Extract journal/conference
        journal = None
        publication_title = ieee_item.get("publication_title")
        if publication_title:
            journal = publication_title
        
        # Build the paper model
        return ResearchPaper(
            paper_id=f"ieee_{article_id}",
            source="ieee",
            source_id=article_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=published_date,
            url=url,
            doi=doi,
            categories=categories,
            pdf_url=pdf_url,
            journal=journal,
            metadata={
                "publisher": ieee_item.get("publisher", ""),
                "publication_year": ieee_item.get("publication_year", ""),
                "content_type": ieee_item.get("content_type", ""),
                "access_type": ieee_item.get("access_type", ""),
                "is_open_access": ieee_item.get("is_open_access", False),
                "conference_location": ieee_item.get("conference_location", ""),
                "conference_dates": ieee_item.get("conference_dates", "")
            }
        )
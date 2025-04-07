"""
Semantic Scholar API client for fetching research papers.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
import time

from research_pipeline.apis.base import BaseAPIClient
from research_pipeline.models.research_paper import ResearchPaper, Author


class SemanticScholarClient(BaseAPIClient):
    """Client for interacting with the Semantic Scholar API."""
    
    # Base URL for the Semantic Scholar API
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Fields to retrieve from the API
    FIELDS = [
        "paperId", "externalIds", "url", "title", "abstract", "venue", "year", 
        "publicationDate", "journal", "authors", "fieldsOfStudy", "s2FieldsOfStudy", 
        "openAccessPdf", "publicationTypes", "publicationVenue"
    ]
    
    # CS fields in Semantic Scholar
    CS_FIELDS = [
        "Computer Science"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Semantic Scholar API client.
        
        Args:
            api_key: Semantic Scholar API key (if None, will look for SEMANTIC_SCHOLAR_API_KEY in env)
        """
        super().__init__(name="Semantic Scholar")
        
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        # Set up headers
        headers = {
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["x-api-key"] = self.api_key
        else:
            logger.warning("No Semantic Scholar API key provided. Rate limits will be stricter.")
        
        self.session.headers.update(headers)
    
    def fetch_recent_papers(self, days_back: int = 3, limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers published in the last n days from Semantic Scholar.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to fetch
            
        Returns:
            List of ResearchPaper objects
        """
        logger.info(f"Fetching papers from Semantic Scholar published in the last {days_back} days (limit: {limit})")
        
        # Calculate date range
        date_since = self.get_date_n_days_ago(days_back)
        date_str = date_since.strftime("%Y-%m-%d")
        
        # Semantic Scholar API endpoint for paper search
        search_url = f"{self.BASE_URL}/paper/search"
        
        # Prepare query parameters
        fields_str = ",".join(self.FIELDS)
        params = {
            "query": "computer science", 
            "year": f"{date_since.year}-{datetime.now().year}",
            "fields": fields_str,
            "limit": min(100, limit),  # Semantic Scholar max per page is 100
            "offset": 0
        }
        
        logger.debug(f"Semantic Scholar query params: {params}")
        
        # Fetch papers with pagination
        papers = []
        offset = 0
        
        while len(papers) < limit:
            current_params = {**params, "offset": offset}
            
            try:
                response = self._make_request(
                    url=search_url,
                    params=current_params
                )
                
                data = response.json()
                results = data.get("data", [])
                
                if not results:
                    logger.debug("No more results from Semantic Scholar")
                    break
                
                # Process results - filter for recently published
                for item in results:
                    # Check if it's a CS paper
                    if not self._is_cs_paper(item):
                        continue
                        
                    # Check if it's within our date range
                    if not self._is_within_date_range(item, date_since):
                        continue
                    
                    paper = self._convert_to_model(item)
                    papers.append(paper)
                    
                    # Check if we've reached the limit
                    if len(papers) >= limit:
                        break
                
                # Increment offset for pagination
                offset += len(results)
                
                # Check if there are no more results
                if len(results) < params["limit"]:
                    break
                
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching papers from Semantic Scholar: {str(e)}")
                break
        
        logger.info(f"Successfully fetched {len(papers)} papers from Semantic Scholar")
        return papers
    
    def _is_cs_paper(self, item: Dict[str, Any]) -> bool:
        """
        Check if a paper is a computer science paper.
        
        Args:
            item: Semantic Scholar paper data
            
        Returns:
            True if it's a CS paper, False otherwise
        """
        # Check fields of study
        fields_of_study = item.get("fieldsOfStudy", [])
        if any(field in self.CS_FIELDS for field in fields_of_study):
            return True
            
        # Check S2 fields of study (more detailed)
        s2_fields = item.get("s2FieldsOfStudy", [])
        for field in s2_fields:
            field_name = field.get("category", "")
            if "Computer Science" in field_name:
                return True
                
        return False
    
    def _is_within_date_range(self, item: Dict[str, Any], date_since: datetime) -> bool:
        """
        Check if a paper was published within our date range.
        
        Args:
            item: Semantic Scholar paper data
            date_since: Earliest date to consider
            
        Returns:
            True if within range, False otherwise
        """
        pub_date = item.get("publicationDate")
        if not pub_date:
            return False
            
        try:
            paper_date = datetime.fromisoformat(pub_date)
            return paper_date >= date_since
        except (ValueError, TypeError):
            # If we can't parse the date, use year as fallback
            year = item.get("year")
            if year and int(year) >= date_since.year:
                return True
            return False
    
    def _convert_to_model(self, ss_item: Dict[str, Any]) -> ResearchPaper:
        """
        Convert a Semantic Scholar item to our ResearchPaper model.
        
        Args:
            ss_item: A Semantic Scholar item object
            
        Returns:
            ResearchPaper object
        """
        # Extract IDs
        ss_id = ss_item.get("paperId", "")
        doi = ss_item.get("externalIds", {}).get("DOI")
        
        # Extract title and abstract
        title = ss_item.get("title", "")
        abstract = ss_item.get("abstract", "")
        
        # Extract authors
        authors = []
        for author_data in ss_item.get("authors", []):
            name = author_data.get("name", "")
            
            if name:
                # Semantic Scholar author details don't include affiliation in search results
                # Get authors.authorId, then query the /author/{authorId} endpoint if needed
                authors.append(Author(
                    name=name,
                    affiliation=None
                ))
        
        # Extract publication date
        published_date = None
        pub_date_str = ss_item.get("publicationDate")
        if pub_date_str:
            try:
                published_date = datetime.fromisoformat(pub_date_str)
            except ValueError:
                try:
                    # Try just year-month-day format
                    published_date = datetime.strptime(pub_date_str[:10], "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Could not parse Semantic Scholar date: {pub_date_str}")
        
        # Extract URL
        url = ss_item.get("url")
        
        # Extract PDF URL
        pdf_url = None
        if "openAccessPdf" in ss_item and ss_item["openAccessPdf"]:
            pdf_url = ss_item["openAccessPdf"].get("url")
        
        # Extract categories/fields of study
        categories = ss_item.get("fieldsOfStudy", [])
        # Add s2 fields for more detail
        for field in ss_item.get("s2FieldsOfStudy", []):
            if "category" in field and field["category"] not in categories:
                categories.append(field["category"])
        
        # Extract journal/venue
        journal = None
        venue = ss_item.get("venue")
        if venue:
            journal = venue
        elif "journal" in ss_item and ss_item["journal"]:
            journal = ss_item["journal"].get("name")
        elif "publicationVenue" in ss_item and ss_item["publicationVenue"]:
            journal = ss_item["publicationVenue"].get("name")
        
        # Build the paper model
        return ResearchPaper(
            paper_id=f"semanticscholar_{ss_id}",
            source="semantic_scholar",
            source_id=ss_id,
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
                "year": ss_item.get("year"),
                "citation_count": ss_item.get("citationCount"),
                "reference_count": ss_item.get("referenceCount"),
                "influential_citation_count": ss_item.get("influentialCitationCount"),
                "publication_types": ss_item.get("publicationTypes", [])
            }
        )
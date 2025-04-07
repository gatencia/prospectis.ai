"""
CrossRef API client for fetching research papers.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import requests
import time
import os

from research_pipeline.apis.base import BaseAPIClient
from research_pipeline.models.research_paper import ResearchPaper, Author


class CrossRefClient(BaseAPIClient):
    """Client for interacting with the CrossRef API."""
    
    # Base URL for the CrossRef API
    BASE_URL = "https://api.crossref.org/works"
    
    # CS categories/types in CrossRef
    CS_CATEGORIES = [
        "computer science",
        "artificial intelligence",
        "machine learning",
        "data mining",
        "computer vision",
        "natural language processing",
        "information retrieval",
        "computational linguistics",
        "software engineering",
        "human-computer interaction"
    ]
    
    def __init__(self, email: Optional[str] = None):
        """
        Initialize the CrossRef API client.
        
        Args:
            email: Email to include in User-Agent for better API service (recommended by CrossRef)
        """
        super().__init__(name="CrossRef")
        if email is None:
            email = os.getenv("CROSSREF_EMAIL")  # Load from .env
        
        # Set up headers with polite user-agent
        user_agent = "ProspectisBot/1.0"
        if email:
            user_agent += f" ({email})"
        
        self.session.headers.update({
            "User-Agent": user_agent
        })

    def fetch_recent_papers(self, days_back: int = 3, limit: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers published in the last n days from CrossRef.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of papers to fetch
            
        Returns:
            List of ResearchPaper objects
        """
        logger.info(f"Fetching papers from CrossRef published in the last {days_back} days (limit: {limit})")
        
        # Calculate date range
        date_since = self.get_date_n_days_ago(days_back)
        date_str = date_since.strftime("%Y-%m-%d")
        
        # Prepare query parameters
        params = {
            "filter": f"from-pub-date:{date_str},has-abstract:true",
            "sort": "published",
            "order": "desc",
            "rows": min(limit, 100),  # CrossRef max is 100 per page
            "select": "DOI,title,abstract,author,published-print,published-online,subject,URL,resource,type,container-title"
        }
        
        # Add filter for CS subjects (not perfect, but helps)
        subject_filter = " OR ".join([f"subject:\"{category}\"" for category in self.CS_CATEGORIES])
        params["query"] = subject_filter
        
        logger.debug(f"CrossRef query params: {params}")
        
        # Fetch papers with pagination
        papers = []
        offset = 0
        
        while len(papers) < limit:
            current_params = {**params, "offset": offset}
            
            try:
                response = self._make_request(
                    url=self.BASE_URL,
                    params=current_params
                )
                
                data = response.json()
                results = data.get("message", {}).get("items", [])
                
                if not results:
                    logger.debug("No more results from CrossRef")
                    break
                
                # Process results
                for item in results:
                    # Skip non-CS papers as best we can
                    if not self._is_cs_paper(item):
                        continue
                    
                    paper = self._convert_to_model(item)
                    papers.append(paper)
                    
                    # Check if we've reached the limit
                    if len(papers) >= limit:
                        break
                
                # Increment offset for pagination
                offset += len(results)
                
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching papers from CrossRef: {str(e)}")
                break
        
        logger.info(f"Successfully fetched {len(papers)} papers from CrossRef")
        return papers
    
    def _is_cs_paper(self, item: Dict[str, Any]) -> bool:
        """
        Check if a paper is related to computer science.
        This is a best-effort function since CrossRef doesn't have a perfect categorization.
        
        Args:
            item: CrossRef paper data
            
        Returns:
            True if the paper appears to be CS-related, False otherwise
        """
        # Check subjects
        subjects = item.get("subject", [])
        for subject in subjects:
            subject_lower = subject.lower()
            if any(cs_cat.lower() in subject_lower for cs_cat in self.CS_CATEGORIES):
                return True
                
        # Check title and abstract
        title = " ".join(item.get("title", []))
        abstract = item.get("abstract", "")
        
        text = (title + " " + abstract).lower()
        for cs_term in ["algorithm", "computation", "data", "network", "software", "programming", 
                       "machine learning", "artificial intelligence", "neural network"]:
            if cs_term in text:
                return True
                
        # If it's explicitly a computer science journal/conference, accept it
        container = item.get("container-title", [""])[0].lower()
        cs_containers = ["ieee", "acm", "computer", "computing", "informatics", "software", 
                        "data", "information", "journal of machine learning"]
        
        if any(cs_cont in container for cs_cont in cs_containers):
            return True
            
        return False
    
    def _convert_to_model(self, crossref_item: Dict[str, Any]) -> ResearchPaper:
        """
        Convert a CrossRef item to our ResearchPaper model.
        
        Args:
            crossref_item: A CrossRef item object
            
        Returns:
            ResearchPaper object
        """
        doi = crossref_item.get("DOI")
        
        # Extract title
        title = " ".join(crossref_item.get("title", []))
        
        # Extract abstract
        abstract = crossref_item.get("abstract", "")
        # Clean up HTML in abstract if present
        if abstract and "<" in abstract:
            # Simple HTML tag removal - for better handling use BeautifulSoup
            abstract = abstract.replace("<p>", "").replace("</p>", "\n").replace("<jats:p>", "").replace("</jats:p>", "\n")
        
        # Extract authors
        authors = []
        for author_data in crossref_item.get("author", []):
            name_parts = []
            if "given" in author_data:
                name_parts.append(author_data["given"])
            if "family" in author_data:
                name_parts.append(author_data["family"])
                
            name = " ".join(name_parts).strip()
            if name:
                affiliation = None
                if "affiliation" in author_data and author_data["affiliation"]:
                    aff_names = [aff.get("name", "") for aff in author_data["affiliation"] if "name" in aff]
                    affiliation = "; ".join(aff_names)
                    
                authors.append(Author(
                    name=name,
                    affiliation=affiliation
                ))
        
        # Extract publication date
        published_date = None
        if "published-online" in crossref_item and crossref_item["published-online"]:
            date_parts = crossref_item["published-online"]["date-parts"][0]
            if len(date_parts) >= 3:
                published_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        elif "published-print" in crossref_item and crossref_item["published-print"]:
            date_parts = crossref_item["published-print"]["date-parts"][0]
            if len(date_parts) >= 3:
                published_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        elif "created" in crossref_item and crossref_item["created"]:
            date_parts = crossref_item["created"]["date-parts"][0]
            if len(date_parts) >= 3:
                published_date = datetime(date_parts[0], date_parts[1], date_parts[2])
        
        # Extract URL
        url = crossref_item.get("URL")
        
        # Extract categories/subjects
        categories = crossref_item.get("subject", [])
        
        # Extract journal
        journal = None
        if "container-title" in crossref_item and crossref_item["container-title"]:
            journal = crossref_item["container-title"][0]
        
        # Build the paper model
        return ResearchPaper(
            paper_id=f"crossref_{doi}",
            source="crossref",
            source_id=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            published_date=published_date,
            url=url,
            doi=doi,
            categories=categories,
            pdf_url=None,  # CrossRef doesn't directly provide PDF URLs
            journal=journal,
            metadata={
                "type": crossref_item.get("type"),
                "publisher": crossref_item.get("publisher"),
                "issue": crossref_item.get("issue"),
                "volume": crossref_item.get("volume"),
                "page": crossref_item.get("page")
            }
        )
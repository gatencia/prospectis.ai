"""
Main entry point for the research papers data pipeline.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import traceback
from loguru import logger

from research_pipeline.apis.arxiv import ArxivClient
from research_pipeline.apis.crossref import CrossRefClient
from research_pipeline.apis.ieee import IEEEClient
from research_pipeline.apis.semantic_scholar import SemanticScholarClient
from research_pipeline.models.research_paper import ResearchPaper
from research_pipeline.db.connection import get_papers_collection, close_connection
from research_pipeline.utils.logging_config import setup_logging
from research_pipeline.config import PAPERS_FETCH_LIMIT, PAPERS_DAYS_LOOKBACK, SOURCES


class ResearchPipeline:
    """Main pipeline for fetching and processing research papers."""
    
    def __init__(self, sources: Optional[List[str]] = None):
        """
        Initialize the research pipeline.
        
        Args:
            sources: List of sources to use (defaults to all)
        """
        setup_logging()
        logger.info("Initializing research pipeline")
        
        self.sources = sources or SOURCES
        logger.info(f"Using sources: {', '.join(self.sources)}")
        
        # Initialize API clients based on selected sources
        self.clients = {}
        if "arxiv" in self.sources:
            self.clients["arxiv"] = ArxivClient()
        if "crossref" in self.sources:
            self.clients["crossref"] = CrossRefClient()
        if "ieee" in self.sources:
            self.clients["ieee"] = IEEEClient()
        if "semantic_scholar" in self.sources:
            self.clients["semantic_scholar"] = SemanticScholarClient()
    
    def run(self, days_back: int = None, limit: int = None) -> Dict[str, int]:
        """
        Run the pipeline to fetch and store papers.
        
        Args:
            days_back: Number of days to look back (default: from config)
            limit: Maximum number of papers to fetch per source (default: from config)
            
        Returns:
            Dict with sources as keys and number of papers fetched as values
        """
        days_back = days_back or PAPERS_DAYS_LOOKBACK
        limit = limit or PAPERS_FETCH_LIMIT
        
        logger.info(f"Running research pipeline (days_back={days_back}, limit={limit})")
        
        start_time = time.time()
        results = {}
        
        # Process each source
        for source, client in self.clients.items():
            try:
                source_start = time.time()
                logger.info(f"Fetching papers from {source}")
                
                # Fetch papers from source
                papers = client.fetch_recent_papers(days_back=days_back, limit=limit)
                
                # Store papers in database
                stored_count = self._store_papers(papers, source)
                
                # Record results
                results[source] = stored_count
                
                source_end = time.time()
                logger.info(f"Processed {source}: fetched {len(papers)}, stored {stored_count} papers in {source_end - source_start:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing source {source}: {str(e)}")
                logger.error(traceback.format_exc())
                results[source] = 0
        
        # Close API client sessions
        self._close_clients()
        
        end_time = time.time()
        total_papers = sum(results.values())
        logger.info(f"Pipeline completed: processed {total_papers} papers in {end_time - start_time:.2f}s")
        
        return results
    
    def _store_papers(self, papers: List[ResearchPaper], source: str) -> int:
        """
        Store papers in the database.
        
        Args:
            papers: List of papers to store
            source: Source of the papers
            
        Returns:
            Number of papers stored
        """
        if not papers:
            logger.warning(f"No papers to store from {source}")
            return 0
            
        logger.info(f"Storing {len(papers)} papers from {source}")
        
        # Get papers collection
        collection = get_papers_collection()
        
        # Track stats
        new_count = 0
        updated_count = 0
        
        for paper in papers:
            # Convert to dict for storage
            paper_dict = paper.to_dict()
            
            try:
                # Check if paper already exists
                existing_paper = collection.find_one({
                    "source": paper.source,
                    "source_id": paper.source_id
                })
                
                if existing_paper:
                    # Update the paper with new info
                    paper_dict["last_updated"] = datetime.utcnow()
                    collection.update_one(
                        {"_id": existing_paper["_id"]},
                        {"$set": paper_dict}
                    )
                    updated_count += 1
                else:
                    # Insert new paper
                    collection.insert_one(paper_dict)
                    new_count += 1
                    
            except Exception as e:
                logger.error(f"Error storing paper {paper.paper_id}: {str(e)}")
        
        logger.info(f"Stored {new_count} new papers and updated {updated_count} papers from {source}")
        return new_count + updated_count
    
    def _close_clients(self):
        """Close all API client sessions."""
        for source, client in self.clients.items():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing {source} client: {str(e)}")
        
        # Close database connection
        close_connection()
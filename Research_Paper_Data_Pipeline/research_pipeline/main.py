"""
Main entry point for the research papers data pipeline.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
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
from research_pipeline.config import PAPERS_FETCH_LIMIT, PAPERS_DAYS_LOOKBACK, SOURCES, IEEE_API_KEY

class ResearchPipeline:
    """Main pipeline for fetching and processing research papers."""
    
    def __init__(self, sources: Optional[List[str]] = None):
        """Initialize the research pipeline."""
        setup_logging()
        logger.info("Initializing research pipeline")

        # Dynamically adjust sources if IEEE key is missing
        if sources is None:
            sources = SOURCES.copy()
            if "ieee" in sources and not IEEE_API_KEY:
                logger.warning("IEEE API key not found — skipping ieee source.")
                sources.remove("ieee")

        self.sources = sources
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
    
    def run(self, days_back: int, limit: int) -> Tuple[Dict[str, int], int]:
        """
        Run the research paper pipeline.

        Args:
            days_back: Days to look back
            limit: Number of papers per source to attempt fetching

        Returns:
            Tuple:
                - Dictionary of new papers inserted per source
                - Total new papers inserted
        """
        logger.info(f"Running research pipeline (days_back={days_back}, limit={limit})")
        results = {}
        total_new = 0

        for source in self.sources:
            logger.info(f"Fetching papers from {source}")
            client = self.clients.get(source)
            if not client:
                logger.warning(f"No client for source: {source}")
                continue

            try:
                papers = client.fetch_recent_papers(days_back=days_back, limit=limit)
            except Exception as e:
                logger.error(f"Error fetching papers from {source}: {e}")
                papers = []

            new_count = self._store_papers(papers, source)
            results[source] = new_count
            total_new += new_count

            logger.info(f"Processed {source}: stored {new_count} new papers")

        close_connection()
        logger.info(f"Pipeline completed: inserted {total_new} new papers")
        return results, total_new

    def _store_papers(self, papers: List[ResearchPaper], source: str) -> int:
        """
        Store papers in the database.

        Args:
            papers: List of papers to store
            source: Source of the papers

        Returns:
            Number of new papers inserted
        """
        if not papers:
            logger.warning(f"No papers to store from {source}")
            return 0

        logger.info(f"Storing {len(papers)} papers from {source}")
        collection = get_papers_collection()

        new_count = 0

        for paper in papers:
            paper_dict = paper.to_dict()

            try:
                existing_paper = collection.find_one({
                    "source": paper.source,
                    "source_id": paper.source_id
                })

                if existing_paper:
                    paper_dict["last_updated"] = datetime.utcnow()
                    collection.update_one(
                        {"_id": existing_paper["_id"]},
                        {"$set": paper_dict}
                    )
                else:
                    collection.insert_one(paper_dict)
                    new_count += 1

            except Exception as e:
                logger.warning(f"Failed to store paper from {source}: {e}")
                continue

        logger.info(f"Stored {new_count} new papers from {source}")
        return new_count

    def _close_clients(self):
        """Close all API client sessions."""
        for source, client in self.clients.items():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing {source} client: {str(e)}")
        
        # Close database connection
        close_connection()
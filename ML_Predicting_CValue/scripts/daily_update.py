#!/usr/bin/env python3
"""
Daily Update Script for the Prospectis ML System.
Performs daily updates to keep the system current:
1. Updates embeddings for new papers and problems
2. Rebuilds vector indices if needed
3. Updates commercial scores for new papers
4. Updates paper-problem matches
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import components
from embeddings.embed_papers import embed_papers
from embeddings.embed_problems import embed_problems
from models.commercial_value.proxy_label_generator import ProxyLabelGenerator
from models.matching.similarity_model import SimilarityModel
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

# Set up logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"daily_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_embeddings():
    """Generate embeddings for new papers and problems."""
    logger.info("Updating embeddings for new papers...")
    try:
        embed_papers()
    except Exception as e:
        logger.error(f"Error updating paper embeddings: {e}")
    
    logger.info("Updating embeddings for new problems...")
    try:
        embed_problems()
    except Exception as e:
        logger.error(f"Error updating problem embeddings: {e}")


def update_commercial_scores(limit=0):
    """Update commercial value scores for papers without them."""
    logger.info("Updating commercial value scores...")
    try:
        generator = ProxyLabelGenerator()
        results = generator.generate_labels_for_all_papers(limit=limit)
        logger.info(f"Updated commercial scores for {len(results)} papers")
    except Exception as e:
        logger.error(f"Error updating commercial scores: {e}")


def update_paper_problem_matches():
    """Update paper-problem matches."""
    logger.info("Updating paper-problem matches...")
    try:
        similarity_model = SimilarityModel()
        total_papers, total_matches = similarity_model.update_all_matches()
        logger.info(f"Updated matches for {total_papers} papers, found {total_matches} total matches")
    except Exception as e:
        logger.error(f"Error updating paper-problem matches: {e}")


def daily_update(skip_embeddings=False, skip_scores=False, skip_matches=False, limit=0):
    """
    Perform the daily update process.
    
    Args:
        skip_embeddings: Whether to skip embedding updates
        skip_scores: Whether to skip commercial score updates
        skip_matches: Whether to skip match updates
        limit: Limit the number of papers to process (0 for all)
    """
    start_time = time.time()
    logger.info("Starting daily update process")
    
    # Connect to MongoDB
    mongo = get_mongo_connector()
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB. Aborting update.")
        return
    
    # Check if there are new papers/problems
    paper_collection = mongo.get_papers_collection()
    problem_collection = mongo.get_problems_collection()
    
    unembedded_papers = paper_collection.count_documents({"scibert_embedding": {"$exists": False}})
    unembedded_problems = problem_collection.count_documents({"sbert_embedding": {"$exists": False}})
    unscored_papers = paper_collection.count_documents({"commercial_score": {"$exists": False}})
    
    logger.info(f"Found {unembedded_papers} papers without embeddings")
    logger.info(f"Found {unembedded_problems} problems without embeddings")
    logger.info(f"Found {unscored_papers} papers without commercial scores")
    
    # Update embeddings if needed
    if not skip_embeddings and (unembedded_papers > 0 or unembedded_problems > 0):
        update_embeddings()
    elif skip_embeddings:
        logger.info("Skipping embedding updates")
    else:
        logger.info("No new papers/problems to embed")
    
    # Update commercial scores
    if not skip_scores:
        update_commercial_scores(limit)
    else:
        logger.info("Skipping commercial score updates")
    
    # Update paper-problem matches
    if not skip_matches:
        update_paper_problem_matches()
    else:
        logger.info("Skipping paper-problem match updates")
    
    # Log completion
    duration = time.time() - start_time
    logger.info(f"Daily update completed in {duration:.2f} seconds")


def main():
    """Parse arguments and run daily update."""
    parser = argparse.ArgumentParser(description="Run daily updates for the Prospectis ML system")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding updates")
    parser.add_argument("--skip-scores", action="store_true", help="Skip commercial score updates")
    parser.add_argument("--skip-matches", action="store_true", help="Skip paper-problem match updates")
    parser.add_argument("--limit", type=int, default=0, help="Limit papers to process (0 for all)")
    
    args = parser.parse_args()
    
    daily_update(
        skip_embeddings=args.skip_embeddings,
        skip_scores=args.skip_scores,
        skip_matches=args.skip_matches,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
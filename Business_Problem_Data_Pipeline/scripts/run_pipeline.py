#!/usr/bin/env python3
"""Script to run the entire business problems pipeline."""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Business_Problem_Data_Pipeline.config import MONGO_URI, DB_NAME, COLLECTION_NAME
from Business_Problem_Data_Pipeline.db.connection import mongo
from Business_Problem_Data_Pipeline.apis.reddit import RedditClient
from Business_Problem_Data_Pipeline.apis.stack_exchange import StackExchangeClient
from Business_Problem_Data_Pipeline.processors.text_processor import TextProcessor
from Business_Problem_Data_Pipeline.processors.keyword_extractor import KeywordExtractor
from Business_Problem_Data_Pipeline.utils.logging_config import setup_logging

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run the business problems pipeline')
    parser.add_argument('--sources', nargs='+', default=['reddit', 'stack_exchange'],
                        choices=['reddit', 'stack_exchange', 'twitter', 'news', 'adzuna', 'patents', 'hacker_news'],
                        help='Data sources to fetch from')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to look back')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of problems to fetch per source')
    parser.add_argument('--process', action='store_true',
                        help='Process the fetched problems')
    return parser.parse_args()

def run_pipeline(sources, days, limit, process):
    """
    Run the business problems pipeline.
    
    Args:
        sources (list): List of data sources to fetch from
        days (int): Number of days to look back
        limit (int): Maximum number of problems to fetch per source
        process (bool): Whether to process the fetched problems
    """
    logger = logging.getLogger('pipeline')
    logger.info(f"Starting business problems pipeline with sources: {sources}")
    
    # Connect to MongoDB
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    problems = []
    
    # Fetch from Reddit
    if 'reddit' in sources:
        logger.info("Fetching from Reddit")
        reddit_client = RedditClient()
        reddit_problems = reddit_client.fetch_problems(time_filter="day", limit=limit)
        problems.extend(reddit_problems)
        logger.info(f"Fetched {len(reddit_problems)} problems from Reddit")
    
    # Fetch from Stack Exchange
    if 'stack_exchange' in sources:
        logger.info("Fetching from Stack Exchange")
        stack_client = StackExchangeClient()
        stack_problems = stack_client.fetch_problems(days=days, limit=limit)
        problems.extend(stack_problems)
        logger.info(f"Fetched {len(stack_problems)} problems from Stack Exchange")
    
    # Process problems if requested
    if process and problems:
        logger.info("Processing problems")
        text_processor = TextProcessor()
        keyword_extractor = KeywordExtractor()
        
        for problem in problems:
            # Clean text
            problem = text_processor.process(problem)
            # Extract keywords
            problem = keyword_extractor.extract_keywords(problem)
    
    # Save problems to MongoDB
    if problems:
        logger.info(f"Saving {len(problems)} problems to MongoDB")
        collection = mongo.get_collection()
        
        for problem in problems:
            # Convert to dictionary
            problem_dict = problem.to_dict()
            
            # Check if problem already exists
            existing = collection.find_one({"source": problem.source, "source_id": problem.source_id})
            
            if existing:
                # Update existing problem
                collection.update_one(
                    {"source": problem.source, "source_id": problem.source_id},
                    {"$set": problem_dict}
                )
                logger.debug(f"Updated problem {problem.id} in MongoDB")
            else:
                # Insert new problem
                collection.insert_one(problem_dict)
                logger.debug(f"Inserted problem {problem.id} into MongoDB")
        
        logger.info(f"Saved {len(problems)} problems to MongoDB")
    else:
        logger.warning("No problems fetched")
    
    # Close connections
    mongo.close()
    logger.info("Pipeline completed")

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging('pipeline')
    
    # Parse arguments
    args = parse_args()
    
    # Run pipeline
    try:
        run_pipeline(args.sources, args.days, args.limit, args.process)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
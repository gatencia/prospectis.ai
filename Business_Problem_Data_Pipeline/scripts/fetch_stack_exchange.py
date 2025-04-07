#!/usr/bin/env python3
"""Script to fetch problems from Stack Exchange."""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Business_Problem_Data_Pipeline.db.connection import mongo
from Business_Problem_Data_Pipeline.apis.stack_exchange import StackExchangeClient
from Business_Problem_Data_Pipeline.processors.text_processor import TextProcessor
from Business_Problem_Data_Pipeline.processors.keyword_extractor import KeywordExtractor
from Business_Problem_Data_Pipeline.utils.logging_config import setup_logging
from Business_Problem_Data_Pipeline.config import STACK_EXCHANGE_SITES

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Fetch problems from Stack Exchange')
    parser.add_argument('--sites', nargs='+', default=None,
                        help='Stack Exchange sites to fetch from (default: use config)')
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to look back')
    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of questions to fetch per site')
    parser.add_argument('--process', action='store_true',
                        help='Process the fetched problems')
    return parser.parse_args()

def main():
    """Main function to fetch problems from Stack Exchange."""
    logger = logging.getLogger('stack_exchange')
    logger.info("Starting Stack Exchange fetcher")
    
    # Parse arguments
    args = parse_args()
    
    # Connect to MongoDB
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Initialize clients and processors
    stack_client = StackExchangeClient()
    text_processor = TextProcessor() if args.process else None
    keyword_extractor = KeywordExtractor() if args.process else None
    
    # Fetch problems
    logger.info(f"Fetching problems from Stack Exchange from the last {args.days} days")
    problems = stack_client.fetch_problems(
        sites=args.sites,
        days=args.days,
        limit=args.limit
    )
    
    if not problems:
        logger.warning("No problems fetched from Stack Exchange")
        mongo.close()
        return
    
    logger.info(f"Fetched {len(problems)} problems from Stack Exchange")
    
    # Process problems if requested
    if args.process:
        logger.info("Processing problems")
        for problem in problems:
            problem = text_processor.process(problem)
            problem = keyword_extractor.extract_keywords(problem)
    
    # Save problems to MongoDB
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
        else:
            # Insert new problem
            collection.insert_one(problem_dict)
    
    logger.info(f"Saved {len(problems)} problems to MongoDB")
    
    # Close connections
    mongo.close()
    logger.info("Stack Exchange fetcher completed")

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging('stack_exchange')
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Stack Exchange fetcher failed: {e}")
        sys.exit(1)
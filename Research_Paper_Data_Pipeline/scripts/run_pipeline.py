#!/usr/bin/env python
"""
Script to run the research pipeline once.
"""

import sys
import argparse
from pathlib import Path
from pymongo import MongoClient


# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research_pipeline.main import ResearchPipeline
from research_pipeline.utils.logging_config import setup_logging
from research_pipeline.config import PAPERS_DAYS_LOOKBACK, PAPERS_FETCH_LIMIT, SOURCES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the research papers pipeline")
    
    parser.add_argument(
        "--days", 
        type=int, 
        default=PAPERS_DAYS_LOOKBACK,
        help=f"Number of days to look back (default: {PAPERS_DAYS_LOOKBACK})"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=PAPERS_FETCH_LIMIT,
        help=f"Maximum number of papers to fetch per source (default: {PAPERS_FETCH_LIMIT})"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=SOURCES,
        default=SOURCES,
        help=f"Sources to fetch papers from (default: {' '.join(SOURCES)})"
    )
    
    return parser.parse_args()


def main():
    """Run the pipeline with command line arguments."""
    setup_logging()
    args = parse_args()

    pipeline = ResearchPipeline(sources=args.sources)
    total_papers = 0
    days_back = args.days
    attempt = 0
    final_results = {}

    while total_papers < 500 and attempt < 5:  # goal: 500 new papers
        print(f"\n🔁 Attempt {attempt + 1} — Looking back {days_back} days")
        
        # Fetch and store new papers only
        results, new_this_round = pipeline.run(days_back=days_back, limit=args.limit)
        total_papers += new_this_round

        # Merge source-specific counts into final_results
        for source, count in results.items():
            final_results[source] = final_results.get(source, 0) + count

        if total_papers >= 500:
            break

        days_back += 30  # Expand time window
        attempt += 1

    print("\n📊 Final Pipeline Results:")
    print("---------------------------")
    for source, count in final_results.items():
        print(f"{source}: {count} new papers")
    print(f"Total new papers inserted: {total_papers}")
    client = MongoClient("mongodb://localhost:27017")
    db = client["prospectis"]
    collection = db["research_papers"]

    print("Total documents:", collection.count_documents({}))
    print("Distinct sources:", collection.distinct("source"))

if __name__ == "__main__":
    main()
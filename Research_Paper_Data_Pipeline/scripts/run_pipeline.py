#!/usr/bin/env python
"""
Script to run the research pipeline once.
"""

import sys
import argparse
from pathlib import Path

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
    results = pipeline.run(days_back=args.days, limit=args.limit)
    
    # Print summary of results
    print("\nPipeline Results:")
    print("-----------------")
    total = 0
    for source, count in results.items():
        print(f"{source}: {count} papers")
        total += count
    print(f"Total: {total} papers")


if __name__ == "__main__":
    main()
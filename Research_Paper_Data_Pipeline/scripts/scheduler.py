#!/usr/bin/env python
"""
Scheduler for running the research pipeline at regular intervals.
"""

import sys
import time
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import schedule
from loguru import logger

from research_pipeline.main import ResearchPipeline
from research_pipeline.utils.logging_config import setup_logging
from research_pipeline.config import (
    PAPERS_DAYS_LOOKBACK, 
    PAPERS_FETCH_LIMIT, 
    SOURCES,
    SCHEDULER_INTERVAL_HOURS
)


# Setup global variables
running = True
pipeline = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Schedule regular runs of the research pipeline")
    
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
    
    parser.add_argument(
        "--interval",
        type=int,
        default=SCHEDULER_INTERVAL_HOURS,
        help=f"Interval between runs in hours (default: {SCHEDULER_INTERVAL_HOURS})"
    )
    
    return parser.parse_args()


def signal_handler(sig, frame):
    """Handle signal interrupts gracefully."""
    global running
    logger.info("Received shutdown signal. Stopping scheduler...")
    running = False


def run_pipeline_job(days_back, limit, sources):
    """Job function to run the pipeline."""
    global pipeline
    
    logger.info("Starting scheduled pipeline run")
    
    if pipeline is None:
        pipeline = ResearchPipeline(sources=sources)
    
    try:
        results = pipeline.run(days_back=days_back, limit=limit)
        
        # Log results
        total = sum(results.values())
        logger.info(f"Pipeline run completed: processed {total} papers")
        
        return schedule.CancelJob
    except Exception as e:
        logger.error(f"Error in scheduled pipeline run: {str(e)}")
        return schedule.CancelJob


def main():
    """Run the scheduler with command line arguments."""
    setup_logging()
    args = parse_args()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize pipeline
    global pipeline
    pipeline = ResearchPipeline(sources=args.sources)
    
    # Schedule the job to run at the specified interval
    logger.info(f"Setting up scheduler to run every {args.interval} hours")
    
    # Run immediately on startup
    logger.info("Running initial pipeline job")
    run_pipeline_job(args.days, args.limit, args.sources)
    
    # Schedule future runs
    schedule.every(args.interval).hours.do(
        run_pipeline_job, 
        days_back=args.days, 
        limit=args.limit, 
        sources=args.sources
    )
    
    # Calculate and log next run time
    next_run = datetime.now() + timedelta(hours=args.interval)
    logger.info(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run the scheduler loop
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    while running:
        schedule.run_pending()
        time.sleep(1)
    
    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
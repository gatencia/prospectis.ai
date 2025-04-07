#!/usr/bin/env python3
"""Scheduler for regular data collection."""

import os
import sys
import time
import logging
import schedule
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Business_Problem_Data_Pipeline.utils.logging_config import setup_logging

def run_reddit_pipeline():
    """Run the Reddit pipeline."""
    logger.info("Running Reddit pipeline")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'fetch_reddit.py'), '--process']
    subprocess.run(cmd)

def run_stack_exchange_pipeline():
    """Run the Stack Exchange pipeline."""
    logger.info("Running Stack Exchange pipeline")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'fetch_stack_exchange.py'), '--process']
    subprocess.run(cmd)

def run_full_pipeline():
    """Run the full pipeline with all sources."""
    logger.info("Running full pipeline")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'run_pipeline.py'), 
           '--sources', 'reddit', 'stack_exchange', '--process']
    subprocess.run(cmd)

if __name__ == '__main__':
    # Set up logging
    logger = setup_logging('scheduler')
    
    logger.info("Starting scheduler")
    
    # Schedule Reddit pipeline to run every 2 hours
    schedule.every(2).hours.do(run_reddit_pipeline)
    
    # Schedule Stack Exchange pipeline to run every 4 hours
    schedule.every(4).hours.do(run_stack_exchange_pipeline)
    
    # Schedule full pipeline to run daily at midnight
    schedule.every().day.at("00:00").do(run_full_pipeline)
    
    logger.info("Scheduler started")
    
    # Run jobs continuously
    while True:
        schedule.run_pending()
        time.sleep(60)
"""
MongoDB connection handling module for Prospectis research pipeline.
"""

import os
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from loguru import logger

# Global client instance
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """
    Return a MongoDB client instance, creating one if it doesn't exist.
    Uses connection pooling by default.
    
    Returns:
        MongoClient: MongoDB client instance
    """
    global _mongo_client
    
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        logger.debug(f"Connecting to MongoDB at {mongo_uri}")
        _mongo_client = MongoClient(mongo_uri)
        # Test connection
        _mongo_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
    
    return _mongo_client


def get_db() -> Database:
    """
    Get the MongoDB database instance.
    
    Returns:
        Database: MongoDB database instance
    """
    db_name = os.getenv("MONGODB_DB", "prospectis")
    return get_mongo_client()[db_name]


def get_papers_collection() -> Collection:
    """
    Get the research papers collection.
    
    Returns:
        Collection: MongoDB collection for research papers
    """
    collection_name = os.getenv("MONGODB_COLLECTION_PAPERS", "research_papers")
    return get_db()[collection_name]


def close_connection():
    """Close the MongoDB connection if it exists."""
    global _mongo_client
    
    if _mongo_client is not None:
        logger.debug("Closing MongoDB connection")
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB connection closed")
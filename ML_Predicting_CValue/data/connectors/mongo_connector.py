"""
MongoDB connector for Prospectis ML Commercial Value Prediction.
Provides utility functions for connecting to and interacting with MongoDB.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.db_config import MONGO_URI, DB_NAME, PAPERS_COLLECTION, PROBLEMS_COLLECTION, FEEDBACK_COLLECTION

logger = logging.getLogger(__name__)

class MongoConnector:
    """Class to handle MongoDB connections and operations for Prospectis."""
    
    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        """Initialize MongoDB connector.
        
        Args:
            uri: MongoDB connection string
            db_name: Database name
        """
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        
    def connect(self) -> bool:
        """Connect to MongoDB.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.client = MongoClient(self.uri)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB: {self.db_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """Get a collection by name.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Collection or None if not connected
        """
        if not self.db:
            if not self.connect():
                return None
        return self.db[collection_name]
    
    def get_papers_collection(self) -> Optional[Collection]:
        """Get the research papers collection.
        
        Returns:
            Collection or None if not connected
        """
        return self.get_collection(PAPERS_COLLECTION)
    
    def get_problems_collection(self) -> Optional[Collection]:
        """Get the business problems collection.
        
        Returns:
            Collection or None if not connected
        """
        return self.get_collection(PROBLEMS_COLLECTION)
    
    def get_feedback_collection(self) -> Optional[Collection]:
        """Get the user feedback collection.
        
        Returns:
            Collection or None if not connected
        """
        return self.get_collection(FEEDBACK_COLLECTION)
    
    def find_papers(self, query: Dict = None, projection: Dict = None, 
                   limit: int = 0, sort_by: List = None) -> List[Dict]:
        """Find research papers matching a query.
        
        Args:
            query: MongoDB query
            projection: Fields to include/exclude
            limit: Maximum number of results (0 for all)
            sort_by: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching papers
        """
        collection = self.get_papers_collection()
        if not collection:
            return []
            
        cursor = collection.find(query or {}, projection or {})
        
        if limit > 0:
            cursor = cursor.limit(limit)
            
        if sort_by:
            cursor = cursor.sort(sort_by)
            
        return list(cursor)
    
    def find_problems(self, query: Dict = None, projection: Dict = None, 
                     limit: int = 0, sort_by: List = None) -> List[Dict]:
        """Find business problems matching a query.
        
        Args:
            query: MongoDB query
            projection: Fields to include/exclude
            limit: Maximum number of results (0 for all)
            sort_by: List of (field, direction) tuples for sorting
            
        Returns:
            List of matching problems
        """
        collection = self.get_problems_collection()
        if not collection:
            return []
            
        cursor = collection.find(query or {}, projection or {})
        
        if limit > 0:
            cursor = cursor.limit(limit)
            
        if sort_by:
            cursor = cursor.sort(sort_by)
            
        return list(cursor)
    
    def update_paper(self, paper_id: Any, update: Dict) -> bool:
        """Update a paper document.
        
        Args:
            paper_id: Paper document ID
            update: Update operation document
            
        Returns:
            bool: True if update was successful
        """
        collection = self.get_papers_collection()
        if not collection:
            return False
            
        try:
            result = collection.update_one({"_id": paper_id}, update)
            return result.modified_count > 0
        except OperationFailure as e:
            logger.error(f"Failed to update paper {paper_id}: {e}")
            return False
    
    def create_indices(self) -> bool:
        """Create indices for better query performance.
        
        Returns:
            bool: True if successful
        """
        try:
            # Papers collection indices
            papers = self.get_papers_collection()
            if papers:
                papers.create_index([("title", "text"), ("abstract", "text")])
                papers.create_index([("published_date", DESCENDING)])
                papers.create_index([("commercial_score", DESCENDING)])
                
            # Problems collection indices
            problems = self.get_problems_collection()
            if problems:
                problems.create_index([("text", "text")])
                problems.create_index([("source", ASCENDING)])
                problems.create_index([("created_date", DESCENDING)])
                
            # Feedback collection indices
            feedback = self.get_feedback_collection()
            if feedback:
                feedback.create_index([("paper_id", ASCENDING)])
                feedback.create_index([("problem_id", ASCENDING)])
                
            logger.info("Created MongoDB indices")
            return True
        except OperationFailure as e:
            logger.error(f"Failed to create indices: {e}")
            return False
    
    def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None


# Singleton instance for reuse
_connector = None

def get_connector() -> MongoConnector:
    """Get the singleton MongoDB connector instance.
    
    Returns:
        MongoConnector instance
    """
    global _connector
    if _connector is None:
        _connector = MongoConnector()
    return _connector
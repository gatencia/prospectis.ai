"""MongoDB connection handling for business problems pipeline."""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from Business_Problem_Data_Pipeline.config import MONGO_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

class MongoConnection:
    """MongoDB connection manager."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one connection is created."""
        if cls._instance is None:
            cls._instance = super(MongoConnection, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.db = None
            cls._instance.collection = None
        return cls._instance
    
    def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Check if connection is successful
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            logger.info(f"Connected to MongoDB at {MONGO_URI}, using database {DB_NAME}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def get_collection(self):
        """Get the business problems collection."""
        if not self.client:
            self.connect()
        return self.collection
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.collection = None
            logger.info("MongoDB connection closed")

# Create a global instance
mongo = MongoConnection()
# db_config.py
# Placeholder file for the Prospectis ML Commercial Value Prediction project
"""
Database configuration for Prospectis ML Commercial Value Prediction.
Contains connection parameters for MongoDB and vector databases.
"""

# MongoDB configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "prospectis"

# Collection names
PAPERS_COLLECTION = "research_papers"
PROBLEMS_COLLECTION = "business_problems"
FEEDBACK_COLLECTION = "user_feedback"
MATCHES_COLLECTION = "paper_problem_matches"

# Vector database configuration (for FAISS or similar)
VECTOR_INDEX_DIR = "./vector_indices"
PAPER_INDEX_NAME = "paper_embeddings"
PROBLEM_INDEX_NAME = "problem_embeddings"

# Embedding dimensions
EMBEDDING_DIM = 768  # For SciBERT/SBERT

# Batch sizes for processing
BATCH_SIZE = 16
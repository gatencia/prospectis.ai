# model_config.py
# Placeholder file for the Prospectis ML Commercial Value Prediction project
"""
Model configuration for Prospectis ML Commercial Value Prediction.
Contains parameters for embedding models, training, and scoring.
"""

# Embedding model configurations
SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LENGTH = 512

# Commercial value model parameters
CV_MODEL_TYPE = "xgboost"  # Options: "xgboost", "neural", "lightgbm", "ensemble"
CV_MODEL_PATH = "./models/saved/cv_model.pkl"

# Weak supervision parameters
MIN_PATENT_CITATIONS = 2  # Minimum patent citations for "high" commercial value
MIN_SIMILARITY_SCORE = 0.75  # Minimum similarity score threshold
MIN_PROBLEM_MATCHES = 3  # Minimum number of problems with high similarity

# Similarity search parameters
TOP_K_PROBLEMS = 10  # Number of top problems to fetch per paper
TOP_K_PAPERS = 10  # Number of top papers to fetch per problem
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity score to consider a match

# Feature extraction parameters
USE_PATENT_CITATIONS = True
USE_INDUSTRY_MENTIONS = True
USE_PROBLEM_SIMILARITY = True
USE_AUTHOR_AFFILIATION = True
USE_LLM_SCORING = False  # Optional feature using LLM for scoring

# Training parameters
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.01
EARLY_STOPPING_ROUNDS = 50

# Evaluation metrics
METRICS = ["precision", "recall", "f1", "average_precision", "ndcg"]
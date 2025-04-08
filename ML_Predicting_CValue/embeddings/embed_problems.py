#!/usr/bin/env python3
"""
Embedding generator for business problems.
Generates embeddings for business problems using Sentence-BERT.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import logging

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.db_config import MONGO_URI, DB_NAME, PROBLEMS_COLLECTION, BATCH_SIZE
from config.model_config import SBERT_MODEL, MAX_SEQ_LENGTH
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_FIELD = "sbert_embedding"

def load_model():
    """Load the SBERT model for problem embeddings.
    
    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading model: {SBERT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL)
    model = AutoModel.from_pretrained(SBERT_MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model

def get_embedding(texts, tokenizer, model):
    """Generate embeddings for a batch of texts.
    
    Args:
        texts: List of text strings
        tokenizer: SBERT tokenizer
        model: SBERT model
        
    Returns:
        np.ndarray: Embedding vectors
    """
    # Tokenize with padding and truncation
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH
    ).to(DEVICE)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        # Use mean pooling for sentence embeddings
        attention_mask = tokens['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Calculate the mean embedding for each sequence in the batch
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

    return embeddings.cpu().numpy()

def embed_problems():
    """Generate embeddings for all business problems without existing embeddings."""
    # Connect to MongoDB
    mongo = get_mongo_connector()
    if not mongo.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Get problems without embeddings
    collection = mongo.get_problems_collection()
    if not collection:
        logger.error("Failed to get problems collection")
        return
    
    cursor = collection.find({EMBED_FIELD: {"$exists": False}})
    problems = list(cursor)
    
    if not problems:
        logger.info("No new problems to embed")
        return
    
    logger.info(f"Embedding {len(problems)} business problems...")
    
    # Load model
    tokenizer, model = load_model()
    
    # Process in batches
    problem_ids = []
    all_embeddings = []
    
    for i in tqdm(range(0, len(problems), BATCH_SIZE)):
        batch = problems[i:i + BATCH_SIZE]
        texts = []
        batch_ids = []

        for p in batch:
            # Combine source and text for context
            source = p.get("source", "")
            problem_text = p.get("text", "")
            
            if not problem_text:
                logger.warning(f"Problem {p['_id']} has no text, skipping")
                continue
                
            # Format: [Source] Problem text
            combined = f"[{source}] {problem_text}" if source else problem_text
            texts.append(combined)
            batch_ids.append(p["_id"])

        if not texts:
            continue

        # Generate embeddings
        embeddings = get_embedding(texts, tokenizer, model)
        
        # Store embeddings in MongoDB
        for j, p_id in enumerate(batch_ids):
            embedding_vector = embeddings[j].tolist()
            collection.update_one(
                {"_id": p_id},
                {"$set": {EMBED_FIELD: embedding_vector}}
            )
            
            # Save for vector index
            problem_ids.append(p_id)
            all_embeddings.append(embeddings[j])
    
    # Update vector index if we have new embeddings
    if all_embeddings:
        logger.info("Updating vector index with new embeddings")
        vector_db = get_vector_connector()
        embeddings_array = np.array(all_embeddings)
        vector_db.update_problem_index(embeddings_array, problem_ids)
    
    logger.info("✅ Done embedding business problems.")

if __name__ == "__main__":
    embed_problems()
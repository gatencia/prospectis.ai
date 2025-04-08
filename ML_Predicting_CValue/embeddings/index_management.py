#!/usr/bin/env python3
"""
Vector Index Management for Prospectis ML.
Provides functions for building and maintaining vector indices for papers and problems.
Supports rebuilding indices from scratch or incrementally updating them.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import time
import faiss

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.db_config import VECTOR_INDEX_DIR, EMBEDDING_DIM
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorIndexManager:
    """
    Manages vector indices for embedding-based searches.
    """
    
    def __init__(self, index_dir: str = VECTOR_INDEX_DIR):
        """
        Initialize vector index manager.
        
        Args:
            index_dir: Directory to store vector indices
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        self.vector_db = get_vector_connector()
    
    def rebuild_paper_index(self, use_gpu: bool = False, batch_size: int = 10000) -> bool:
        """
        Rebuild the paper embedding index from scratch.
        
        Args:
            use_gpu: Whether to use GPU for index building
            batch_size: Batch size for processing
            
        Returns:
            bool: True if successful
        """
        logger.info("Rebuilding paper embedding index from scratch")
        start_time = time.time()
        
        # Get all papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        
        # Count total papers with embeddings
        total_papers = paper_collection.count_documents({"scibert_embedding": {"$exists": True}})
        logger.info(f"Found {total_papers} papers with embeddings")
        
        if total_papers == 0:
            logger.warning("No papers with embeddings found. Run embed_papers.py first.")
            return False
        
        # Process in batches
        batch_start = 0
        paper_ids = []
        all_embeddings = []
        
        while batch_start < total_papers:
            logger.info(f"Processing papers {batch_start} to {batch_start + batch_size}")
            
            # Get batch of papers
            cursor = paper_collection.find(
                {"scibert_embedding": {"$exists": True}},
                {"_id": 1, "scibert_embedding": 1}
            ).skip(batch_start).limit(batch_size)
            
            batch_papers = list(cursor)
            
            # Extract embeddings and IDs
            for paper in batch_papers:
                embedding = paper.get("scibert_embedding")
                if embedding and len(embedding) == EMBEDDING_DIM:
                    all_embeddings.append(embedding)
                    paper_ids.append(paper["_id"])
            
            batch_start += batch_size
        
        # Convert to numpy array
        if not all_embeddings:
            logger.error("No valid embeddings found")
            return False
            
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create index using vector connector
        success = self.vector_db.create_paper_index(embeddings_array, paper_ids)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Paper index rebuilt with {len(paper_ids)} vectors in {elapsed_time:.2f} seconds")
        
        return success
    
    def rebuild_problem_index(self, use_gpu: bool = False, batch_size: int = 10000) -> bool:
        """
        Rebuild the problem embedding index from scratch.
        
        Args:
            use_gpu: Whether to use GPU for index building
            batch_size: Batch size for processing
            
        Returns:
            bool: True if successful
        """
        logger.info("Rebuilding problem embedding index from scratch")
        start_time = time.time()
        
        # Get all problems with embeddings
        problem_collection = self.mongo.get_problems_collection()
        
        # Count total problems with embeddings
        total_problems = problem_collection.count_documents({"sbert_embedding": {"$exists": True}})
        logger.info(f"Found {total_problems} problems with embeddings")
        
        if total_problems == 0:
            logger.warning("No problems with embeddings found. Run embed_problems.py first.")
            return False
        
        # Process in batches
        batch_start = 0
        problem_ids = []
        all_embeddings = []
        
        while batch_start < total_problems:
            logger.info(f"Processing problems {batch_start} to {batch_start + batch_size}")
            
            # Get batch of problems
            cursor = problem_collection.find(
                {"sbert_embedding": {"$exists": True}},
                {"_id": 1, "sbert_embedding": 1}
            ).skip(batch_start).limit(batch_size)
            
            batch_problems = list(cursor)
            
            # Extract embeddings and IDs
            for problem in batch_problems:
                embedding = problem.get("sbert_embedding")
                if embedding and len(embedding) == EMBEDDING_DIM:
                    all_embeddings.append(embedding)
                    problem_ids.append(problem["_id"])
            
            batch_start += batch_size
        
        # Convert to numpy array
        if not all_embeddings:
            logger.error("No valid embeddings found")
            return False
            
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Create index using vector connector
        success = self.vector_db.create_problem_index(embeddings_array, problem_ids)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Problem index rebuilt with {len(problem_ids)} vectors in {elapsed_time:.2f} seconds")
        
        return success
    
    def update_paper_index(self, days: int = 1) -> bool:
        """
        Update the paper index with recently added papers.
        
        Args:
            days: Only process papers added/updated in the last N days
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Updating paper index with papers from the last {days} days")
        
        # Calculate the date threshold
        from datetime import datetime, timedelta
        threshold_date = datetime.now() - timedelta(days=days)
        
        # Get recent papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        
        # Query for recent papers with embeddings
        query = {
            "scibert_embedding": {"$exists": True},
            "$or": [
                {"created_date": {"$gte": threshold_date}},
                {"last_updated": {"$gte": threshold_date}}
            ]
        }
        
        cursor = paper_collection.find(query, {"_id": 1, "scibert_embedding": 1})
        papers = list(cursor)
        
        logger.info(f"Found {len(papers)} new/updated papers")
        
        if not papers:
            logger.info("No new papers to add to the index")
            return True
        
        # Extract embeddings and IDs
        paper_ids = []
        embeddings = []
        
        for paper in papers:
            embedding = paper.get("scibert_embedding")
            if embedding and len(embedding) == EMBEDDING_DIM:
                embeddings.append(embedding)
                paper_ids.append(paper["_id"])
        
        # Update index
        embeddings_array = np.array(embeddings).astype('float32')
        success = self.vector_db.update_paper_index(embeddings_array, paper_ids)
        
        logger.info(f"Paper index updated with {len(paper_ids)} new vectors")
        return success
    
    def update_problem_index(self, days: int = 1) -> bool:
        """
        Update the problem index with recently added problems.
        
        Args:
            days: Only process problems added/updated in the last N days
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Updating problem index with problems from the last {days} days")
        
        # Calculate the date threshold
        from datetime import datetime, timedelta
        threshold_date = datetime.now() - timedelta(days=days)
        
        # Get recent problems with embeddings
        problem_collection = self.mongo.get_problems_collection()
        
        # Query for recent problems with embeddings
        query = {
            "sbert_embedding": {"$exists": True},
            "$or": [
                {"created_date": {"$gte": threshold_date}},
                {"last_updated": {"$gte": threshold_date}}
            ]
        }
        
        cursor = problem_collection.find(query, {"_id": 1, "sbert_embedding": 1})
        problems = list(cursor)
        
        logger.info(f"Found {len(problems)} new/updated problems")
        
        if not problems:
            logger.info("No new problems to add to the index")
            return True
        
        # Extract embeddings and IDs
        problem_ids = []
        embeddings = []
        
        for problem in problems:
            embedding = problem.get("sbert_embedding")
            if embedding and len(embedding) == EMBEDDING_DIM:
                embeddings.append(embedding)
                problem_ids.append(problem["_id"])
        
        # Update index
        embeddings_array = np.array(embeddings).astype('float32')
        success = self.vector_db.update_problem_index(embeddings_array, problem_ids)
        
        logger.info(f"Problem index updated with {len(problem_ids)} new vectors")
        return success
    
    def optimize_indices(self, target_nlist: int = 100) -> Tuple[bool, bool]:
        """
        Optimize indices for better search performance.
        Converts flat indices to IVF indices for faster search.
        
        Args:
            target_nlist: Number of clusters for IVF index
            
        Returns:
            tuple: (paper_success, problem_success)
        """
        logger.info("Optimizing vector indices for faster search")
        
        # Load existing indices
        self.vector_db.load_paper_index()
        self.vector_db.load_problem_index()
        
        paper_success = False
        problem_success = False
        
        # Optimize paper index if it exists
        if self.vector_db.paper_index is not None:
            try:
                # Check if already an optimized index
                if isinstance(self.vector_db.paper_index, faiss.IndexIVFFlat):
                    logger.info("Paper index is already optimized (IVF)")
                    paper_success = True
                else:
                    # Get dimensionality and vector count
                    d = self.vector_db.paper_index.d
                    n = self.vector_db.paper_index.ntotal
                    
                    # Determine optimal nlist (number of clusters)
                    nlist = min(target_nlist, int(4 * np.sqrt(n)))
                    nlist = max(nlist, 10)  # At least 10 clusters
                    
                    logger.info(f"Converting paper index to IVF with {nlist} clusters")
                    
                    # Create quantizer
                    quantizer = faiss.IndexFlatL2(d)
                    
                    # Create new optimized index
                    optimized_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                    
                    # Train the index
                    if n < nlist * 39:  # Need enough vectors to train properly
                        logger.warning(f"Not enough vectors to properly train IVF index ({n} < {nlist * 39})")
                        # Use all vectors from the flat index
                        train_vectors = faiss.rev_swig_ptr(self.vector_db.paper_index.get_xb(), n * d).reshape(n, d)
                    else:
                        # Sample vectors for training
                        train_size = min(n, nlist * 39)
                        indices = np.random.choice(n, train_size, replace=False)
                        train_vectors = faiss.rev_swig_ptr(self.vector_db.paper_index.get_xb(), n * d).reshape(n, d)[indices]
                    
                    # Train the index
                    optimized_index.train(train_vectors)
                    
                    # Add vectors from old index
                    optimized_index.add_sa(
                        faiss.rev_swig_ptr(self.vector_db.paper_index.get_xb(), n * d).reshape(n, d)
                    )
                    
                    # Save the optimized index
                    faiss.write_index(optimized_index, str(self.vector_db.paper_index_path))
                    logger.info("Optimized paper index saved")
                    paper_success = True
            except Exception as e:
                logger.error(f"Error optimizing paper index: {e}")
        else:
            logger.warning("Paper index not found. Run rebuild_paper_index first.")
        
        # Optimize problem index if it exists
        if self.vector_db.problem_index is not None:
            try:
                # Check if already an optimized index
                if isinstance(self.vector_db.problem_index, faiss.IndexIVFFlat):
                    logger.info("Problem index is already optimized (IVF)")
                    problem_success = True
                else:
                    # Get dimensionality and vector count
                    d = self.vector_db.problem_index.d
                    n = self.vector_db.problem_index.ntotal
                    
                    if n < 10:
                        logger.warning("Too few problem vectors to optimize. Skipping.")
                        problem_success = True
                    else:
                        # Determine optimal nlist (number of clusters)
                        nlist = min(target_nlist, int(4 * np.sqrt(n)))
                        nlist = max(nlist, 10)  # At least 10 clusters
                        
                        logger.info(f"Converting problem index to IVF with {nlist} clusters")
                        
                        # Create quantizer
                        quantizer = faiss.IndexFlatL2(d)
                        
                        # Create new optimized index
                        optimized_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                        
                        # Train the index
                        if n < nlist * 39:  # Need enough vectors to train properly
                            # Use all vectors from the flat index
                            train_vectors = faiss.rev_swig_ptr(self.vector_db.problem_index.get_xb(), n * d).reshape(n, d)
                        else:
                            # Sample vectors for training
                            train_size = min(n, nlist * 39)
                            indices = np.random.choice(n, train_size, replace=False)
                            train_vectors = faiss.rev_swig_ptr(self.vector_db.problem_index.get_xb(), n * d).reshape(n, d)[indices]
                        
                        # Train the index
                        optimized_index.train(train_vectors)
                        
                        # Add vectors from old index
                        optimized_index.add_sa(
                            faiss.rev_swig_ptr(self.vector_db.problem_index.get_xb(), n * d).reshape(n, d)
                        )
                        
                        # Save the optimized index
                        faiss.write_index(optimized_index, str(self.vector_db.problem_index_path))
                        logger.info("Optimized problem index saved")
                        problem_success = True
            except Exception as e:
                logger.error(f"Error optimizing problem index: {e}")
        else:
            logger.warning("Problem index not found. Run rebuild_problem_index first.")
        
        return paper_success, problem_success


def main():
    """CLI interface for index management."""
    parser = argparse.ArgumentParser(description="Manage vector indices for Prospectis ML")
    parser.add_argument("--rebuild-paper", action="store_true", help="Rebuild paper index from scratch")
    parser.add_argument("--rebuild-problem", action="store_true", help="Rebuild problem index from scratch")
    parser.add_argument("--update-paper", action="store_true", help="Update paper index with recent papers")
    parser.add_argument("--update-problem", action="store_true", help="Update problem index with recent problems")
    parser.add_argument("--optimize", action="store_true", help="Optimize indices for faster search")
    parser.add_argument("--days", type=int, default=1, help="Days to look back for updates")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for index building if available")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for processing")
    parser.add_argument("--nlist", type=int, default=100, help="Number of clusters for IVF optimization")
    
    args = parser.parse_args()
    
    # Create index manager
    manager = VectorIndexManager()
    
    # Execute commands
    if args.rebuild_paper:
        manager.rebuild_paper_index(use_gpu=args.use_gpu, batch_size=args.batch_size)
    
    if args.rebuild_problem:
        manager.rebuild_problem_index(use_gpu=args.use_gpu, batch_size=args.batch_size)
    
    if args.update_paper:
        manager.update_paper_index(days=args.days)
    
    if args.update_problem:
        manager.update_problem_index(days=args.days)
    
    if args.optimize:
        manager.optimize_indices(target_nlist=args.nlist)
    
    # If no arguments, print help
    if not (args.rebuild_paper or args.rebuild_problem or 
            args.update_paper or args.update_problem or args.optimize):
        parser.print_help()


if __name__ == "__main__":
    main()
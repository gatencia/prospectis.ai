"""
Vector database connector for Prospectis ML Commercial Value Prediction.
Provides utility functions for managing vector indices and similarity searches.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import faiss
import pickle
from pathlib import Path

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.db_config import VECTOR_INDEX_DIR, PAPER_INDEX_NAME, PROBLEM_INDEX_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)

class VectorDBConnector:
    """Class to manage vector indices for embedding-based searches."""
    
    def __init__(self, index_dir: str = VECTOR_INDEX_DIR):
        """Initialize vector database connector.
        
        Args:
            index_dir: Directory to store vector indices
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index paths
        self.paper_index_path = self.index_dir / f"{PAPER_INDEX_NAME}.index"
        self.paper_mapping_path = self.index_dir / f"{PAPER_INDEX_NAME}.pkl"
        self.problem_index_path = self.index_dir / f"{PROBLEM_INDEX_NAME}.index"
        self.problem_mapping_path = self.index_dir / f"{PROBLEM_INDEX_NAME}.pkl"
        
        # Initialize indices and mappings
        self.paper_index = None
        self.paper_id_mapping = {}  # Maps vector index to document ID
        self.problem_index = None
        self.problem_id_mapping = {}  # Maps vector index to document ID
    
    def create_paper_index(self, embeddings: np.ndarray, paper_ids: List[Any]) -> bool:
        """Create or replace the paper embeddings index.
        
        Args:
            embeddings: Paper embedding vectors (n_papers, embedding_dim)
            paper_ids: List of paper document IDs corresponding to vectors
            
        Returns:
            bool: True if successful
        """
        if len(embeddings) != len(paper_ids):
            logger.error("Number of embeddings and paper IDs must match")
            return False
            
        try:
            # Create a new index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to the index
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))
            index.add(embeddings)
            
            # Create ID mapping
            id_mapping = {i: paper_id for i, paper_id in enumerate(paper_ids)}
            
            # Save the index
            faiss.write_index(index, str(self.paper_index_path))
            
            # Save the mapping
            with open(self.paper_mapping_path, 'wb') as f:
                pickle.dump(id_mapping, f)
                
            # Update in-memory objects
            self.paper_index = index
            self.paper_id_mapping = id_mapping
            
            logger.info(f"Created paper index with {len(paper_ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create paper index: {e}")
            return False
    
    def create_problem_index(self, embeddings: np.ndarray, problem_ids: List[Any]) -> bool:
        """Create or replace the problem embeddings index.
        
        Args:
            embeddings: Problem embedding vectors (n_problems, embedding_dim)
            problem_ids: List of problem document IDs corresponding to vectors
            
        Returns:
            bool: True if successful
        """
        if len(embeddings) != len(problem_ids):
            logger.error("Number of embeddings and problem IDs must match")
            return False
            
        try:
            # Create a new index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to the index
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))
            index.add(embeddings)
            
            # Create ID mapping
            id_mapping = {i: problem_id for i, problem_id in enumerate(problem_ids)}
            
            # Save the index
            faiss.write_index(index, str(self.problem_index_path))
            
            # Save the mapping
            with open(self.problem_mapping_path, 'wb') as f:
                pickle.dump(id_mapping, f)
                
            # Update in-memory objects
            self.problem_index = index
            self.problem_id_mapping = id_mapping
            
            logger.info(f"Created problem index with {len(problem_ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create problem index: {e}")
            return False
    
    def load_paper_index(self) -> bool:
        """Load the paper embeddings index.
        
        Returns:
            bool: True if successful
        """
        if not self.paper_index_path.exists() or not self.paper_mapping_path.exists():
            logger.warning("Paper index or mapping file does not exist")
            return False
            
        try:
            # Load index
            self.paper_index = faiss.read_index(str(self.paper_index_path))
            
            # Load mapping
            with open(self.paper_mapping_path, 'rb') as f:
                self.paper_id_mapping = pickle.load(f)
                
            logger.info(f"Loaded paper index with {self.paper_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load paper index: {e}")
            return False
    
    def load_problem_index(self) -> bool:
        """Load the problem embeddings index.
        
        Returns:
            bool: True if successful
        """
        if not self.problem_index_path.exists() or not self.problem_mapping_path.exists():
            logger.warning("Problem index or mapping file does not exist")
            return False
            
        try:
            # Load index
            self.problem_index = faiss.read_index(str(self.problem_index_path))
            
            # Load mapping
            with open(self.problem_mapping_path, 'rb') as f:
                self.problem_id_mapping = pickle.load(f)
                
            logger.info(f"Loaded problem index with {self.problem_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load problem index: {e}")
            return False
    
    def find_similar_papers(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[Any, float]]:
        """Find papers similar to a query vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (paper_id, similarity_score) tuples
        """
        if self.paper_index is None:
            if not self.load_paper_index():
                logger.error("Paper index not loaded")
                return []
                
        # Ensure query vector is in correct format
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1).astype('float32'))
        
        # Search for similar vectors
        distances, indices = self.paper_index.search(query_vector, k)
        
        # Convert results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx in self.paper_id_mapping:
                # Convert L2 distance to similarity score (1 - normalized distance)
                similarity = 1.0 - (dist / (2 * max(distances[0])) if max(distances[0]) > 0 else 0)
                results.append((self.paper_id_mapping[idx], similarity))
        
        return results
    
    def find_similar_problems(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[Any, float]]:
        """Find problems similar to a query vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (problem_id, similarity_score) tuples
        """
        if self.problem_index is None:
            if not self.load_problem_index():
                logger.error("Problem index not loaded")
                return []
                
        # Ensure query vector is in correct format
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1).astype('float32'))
        
        # Search for similar vectors
        distances, indices = self.problem_index.search(query_vector, k)
        
        # Convert results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx in self.problem_id_mapping:
                # Convert L2 distance to similarity score (1 - normalized distance)
                similarity = 1.0 - (dist / (2 * max(distances[0])) if max(distances[0]) > 0 else 0)
                results.append((self.problem_id_mapping[idx], similarity))
        
        return results
    
    def update_paper_index(self, new_embeddings: np.ndarray, new_paper_ids: List[Any]) -> bool:
        """Add new papers to the existing index.
        
        Args:
            new_embeddings: New paper embedding vectors to add
            new_paper_ids: List of new paper document IDs
            
        Returns:
            bool: True if successful
        """
        if len(new_embeddings) != len(new_paper_ids):
            logger.error("Number of embeddings and paper IDs must match")
            return False
            
        try:
            # Load existing index if needed
            if self.paper_index is None:
                if not self.load_paper_index():
                    # Create new index if it doesn't exist
                    return self.create_paper_index(new_embeddings, new_paper_ids)
            
            # Add vectors to the index
            new_vectors = np.ascontiguousarray(new_embeddings.astype('float32'))
            start_idx = self.paper_index.ntotal
            self.paper_index.add(new_vectors)
            
            # Update ID mapping
            for i, paper_id in enumerate(new_paper_ids):
                self.paper_id_mapping[start_idx + i] = paper_id
            
            # Save the updated index and mapping
            faiss.write_index(self.paper_index, str(self.paper_index_path))
            with open(self.paper_mapping_path, 'wb') as f:
                pickle.dump(self.paper_id_mapping, f)
                
            logger.info(f"Updated paper index with {len(new_paper_ids)} new vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update paper index: {e}")
            return False
    
    def update_problem_index(self, new_embeddings: np.ndarray, new_problem_ids: List[Any]) -> bool:
        """Add new problems to the existing index.
        
        Args:
            new_embeddings: New problem embedding vectors to add
            new_problem_ids: List of new problem document IDs
            
        Returns:
            bool: True if successful
        """
        if len(new_embeddings) != len(new_problem_ids):
            logger.error("Number of embeddings and problem IDs must match")
            return False
            
        try:
            # Load existing index if needed
            if self.problem_index is None:
                if not self.load_problem_index():
                    # Create new index if it doesn't exist
                    return self.create_problem_index(new_embeddings, new_problem_ids)
            
            # Add vectors to the index
            new_vectors = np.ascontiguousarray(new_embeddings.astype('float32'))
            start_idx = self.problem_index.ntotal
            self.problem_index.add(new_vectors)
            
            # Update ID mapping
            for i, problem_id in enumerate(new_problem_ids):
                self.problem_id_mapping[start_idx + i] = problem_id
            
            # Save the updated index and mapping
            faiss.write_index(self.problem_index, str(self.problem_index_path))
            with open(self.problem_mapping_path, 'wb') as f:
                pickle.dump(self.problem_id_mapping, f)
                
            logger.info(f"Updated problem index with {len(new_problem_ids)} new vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update problem index: {e}")
            return False


# Singleton instance for reuse
_connector = None

def get_connector() -> VectorDBConnector:
    """Get the singleton vector database connector instance.
    
    Returns:
        VectorDBConnector instance
    """
    global _connector
    if _connector is None:
        _connector = VectorDBConnector()
    return _connector
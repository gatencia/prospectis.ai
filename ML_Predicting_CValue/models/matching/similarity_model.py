#!/usr/bin/env python3
"""
Similarity Model for matching papers to business problems.
Provides functions to match papers to problems and vice versa using embeddings.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import TOP_K_PROBLEMS, TOP_K_PAPERS, SIMILARITY_THRESHOLD
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimilarityModel:
    """
    Model for finding similar papers and problems using embedding-based similarity.
    """
    
    def __init__(self):
        """Initialize the similarity model."""
        self.mongo = get_mongo_connector()
        self.vector_db = get_vector_connector()
        
        # Connect to databases
        if not self.mongo.connect():
            logger.error("Failed to connect to MongoDB")
            raise ConnectionError("Could not connect to MongoDB")
            
        # Load vector indices
        self.vector_db.load_paper_index()
        self.vector_db.load_problem_index()
    
    def find_similar_problems(self, paper_id: Any, k: int = TOP_K_PROBLEMS, 
                             threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Find business problems similar to a paper.
        
        Args:
            paper_id: MongoDB ID of the paper
            k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            list: List of similar problems with details
        """
        # Get paper from MongoDB
        paper_collection = self.mongo.get_papers_collection()
        paper = paper_collection.find_one({"_id": paper_id})
        
        if not paper:
            logger.error(f"Paper not found: {paper_id}")
            return []
        
        # Get paper embedding
        embedding = paper.get("scibert_embedding")
        if not embedding:
            logger.error(f"Paper {paper_id} has no embedding")
            return []
        
        # Find similar problems
        similar_problems = self.vector_db.find_similar_problems(
            np.array(embedding),
            k=k + 10  # Get extra to filter by threshold
        )
        
        # Filter by threshold
        similar_problems = [
            (problem_id, score) for problem_id, score in similar_problems
            if score >= threshold
        ][:k]
        
        if not similar_problems:
            return []
        
        # Get problem details
        problem_collection = self.mongo.get_problems_collection()
        result = []
        
        for problem_id, similarity_score in similar_problems:
            problem = problem_collection.find_one({"_id": problem_id})
            if problem:
                result.append({
                    "problem_id": problem_id,
                    "text": problem.get("text", ""),
                    "source": problem.get("source", ""),
                    "created_date": problem.get("created_date", ""),
                    "similarity_score": similarity_score,
                    "url": problem.get("url", "")
                })
        
        return result
    
    def find_similar_papers(self, problem_id: Any, k: int = TOP_K_PAPERS,
                           threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Find papers similar to a business problem.
        
        Args:
            problem_id: MongoDB ID of the problem
            k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            list: List of similar papers with details
        """
        # Get problem from MongoDB
        problem_collection = self.mongo.get_problems_collection()
        problem = problem_collection.find_one({"_id": problem_id})
        
        if not problem:
            logger.error(f"Problem not found: {problem_id}")
            return []
        
        # Get problem embedding
        embedding = problem.get("sbert_embedding")
        if not embedding:
            logger.error(f"Problem {problem_id} has no embedding")
            return []
        
        # Find similar papers
        similar_papers = self.vector_db.find_similar_papers(
            np.array(embedding),
            k=k + 10  # Get extra to filter by threshold
        )
        
        # Filter by threshold
        similar_papers = [
            (paper_id, score) for paper_id, score in similar_papers
            if score >= threshold
        ][:k]
        
        if not similar_papers:
            return []
        
        # Get paper details
        paper_collection = self.mongo.get_papers_collection()
        result = []
        
        for paper_id, similarity_score in similar_papers:
            paper = paper_collection.find_one({"_id": paper_id})
            if paper:
                result.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", {}),
                    "url": paper.get("url", ""),
                    "doi": paper.get("doi", ""),
                    "similarity_score": similarity_score,
                    "commercial_score": paper.get("commercial_score", 0)
                })
        
        # Sort by combined score (similarity * commercial_score)
        for paper in result:
            paper["combined_score"] = paper["similarity_score"] * (1 + paper.get("commercial_score", 0))
        
        result.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return result
    
    def find_papers_for_text(self, problem_text: str, k: int = TOP_K_PAPERS,
                            threshold: float = SIMILARITY_THRESHOLD) -> List[Dict]:
        """
        Find papers similar to a problem text (without storing the problem).
        
        Args:
            problem_text: Problem description text
            k: Number of results to return
            threshold: Minimum similarity score
            
        Returns:
            list: List of similar papers with details
        """
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # Load SBERT model (same as used for problems)
        from config.model_config import SBERT_MODEL, MAX_SEQ_LENGTH
        
        tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL)
        model = AutoModel.from_pretrained(SBERT_MODEL)
        model.eval()
        
        # Generate embedding
        tokens = tokenizer(
            problem_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH
        )
        
        with torch.no_grad():
            outputs = model(**tokens)
            
            # Mean pooling
            attention_mask = tokens['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        # Convert to numpy
        embedding = embedding.numpy()[0]
        
        # Find similar papers
        similar_papers = self.vector_db.find_similar_papers(embedding, k=k + 10)
        
        # Filter by threshold
        similar_papers = [
            (paper_id, score) for paper_id, score in similar_papers
            if score >= threshold
        ][:k]
        
        if not similar_papers:
            return []
        
        # Get paper details
        paper_collection = self.mongo.get_papers_collection()
        result = []
        
        for paper_id, similarity_score in similar_papers:
            paper = paper_collection.find_one({"_id": paper_id})
            if paper:
                result.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", {}),
                    "url": paper.get("url", ""),
                    "doi": paper.get("doi", ""),
                    "similarity_score": similarity_score,
                    "commercial_score": paper.get("commercial_score", 0)
                })
        
        # Sort by combined score (similarity * commercial_score)
        for paper in result:
            paper["combined_score"] = paper["similarity_score"] * (1 + paper.get("commercial_score", 0))
        
        result.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return result
    
    def update_all_matches(self, batch_size: int = 100) -> Tuple[int, int]:
        """
        Update all paper-problem matches in the database.
        Computes and stores top problem matches for each paper.
        
        Args:
            batch_size: Number of papers to process in each batch
            
        Returns:
            tuple: (total_papers, total_matches)
        """
        # Get all papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        cursor = paper_collection.find({"scibert_embedding": {"$exists": True}})
        
        total_papers = 0
        total_matches = 0
        
        # Process in batches
        batch = []
        for paper in cursor:
            batch.append(paper)
            
            if len(batch) >= batch_size:
                papers_processed, matches_found = self._process_paper_batch(batch)
                total_papers += papers_processed
                total_matches += matches_found
                batch = []
                
                logger.info(f"Processed {total_papers} papers, found {total_matches} matches")
        
        # Process remaining papers
        if batch:
            papers_processed, matches_found = self._process_paper_batch(batch)
            total_papers += papers_processed
            total_matches += matches_found
        
        logger.info(f"Completed match update: {total_papers} papers, {total_matches} matches")
        return total_papers, total_matches
    
    def _process_paper_batch(self, papers: List[Dict]) -> Tuple[int, int]:
        """
        Process a batch of papers for matching.
        
        Args:
            papers: List of paper documents
            
        Returns:
            tuple: (papers_processed, matches_found)
        """
        paper_collection = self.mongo.get_papers_collection()
        
        papers_processed = 0
        total_matches = 0
        
        for paper in tqdm(papers, desc="Matching papers to problems"):
            # Find matching problems
            matching_problems = self.find_similar_problems(
                paper["_id"],
                k=TOP_K_PROBLEMS,
                threshold=SIMILARITY_THRESHOLD
            )
            
            # Update paper with matching problems
            if matching_problems:
                paper_collection.update_one(
                    {"_id": paper["_id"]},
                    {"$set": {"matching_problems": matching_problems}}
                )
                
                total_matches += len(matching_problems)
            
            papers_processed += 1
        
        return papers_processed, total_matches


def main():
    """Main function to update all paper-problem matches."""
    similarity_model = SimilarityModel()
    total_papers, total_matches = similarity_model.update_all_matches()
    print(f"Processed {total_papers} papers, found {total_matches} matches")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Cross-Encoder for reranking paper-problem matches.
Provides more precise relevance scoring for paper-problem pairs.
"""

import os
import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn.functional as F

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import SIMILARITY_THRESHOLD
from data.connectors.mongo_connector import get_connector as get_mongo_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossEncoder:
    """
    Cross-encoder model for precise paper-problem relevance scoring.
    """
    
    def __init__(self, model_name="cross-encoder/stsb-roberta-base", device=None):
        """
        Initialize the cross-encoder.
        
        Args:
            model_name: Name of the pre-trained model
            device: Compute device (None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get number of labels from model config
        self.num_labels = self.model.config.num_labels
        
        # Connect to MongoDB
        self.mongo = get_mongo_connector()
        self.mongo.connect()
    
    def score_pair(self, paper_text: str, problem_text: str) -> float:
        """
        Score a single paper-problem pair.
        
        Args:
            paper_text: Paper title and abstract
            problem_text: Problem description
            
        Returns:
            float: Relevance score
        """
        # Prepare input
        features = self.tokenizer(
            paper_text, 
            problem_text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Score
        with torch.no_grad():
            outputs = self.model(**features)
            
            # Handle regression or classification models
            if self.num_labels == 1:  # Regression
                score = outputs.logits.item()
                # Normalize to 0-1 range if needed (model dependent)
                score = (score + 1) / 2  # STS models output -1 to 1
            else:  # Classification
                probs = F.softmax(outputs.logits, dim=1)
                # For binary classification, use positive class probability
                # For multi-class, use weighted average
                if self.num_labels == 2:
                    score = probs[0][1].item()
                else:
                    # Assuming the labels represent scores (e.g., 0-5)
                    scores = torch.arange(self.num_labels, device=self.device).float()
                    score = (probs[0] * scores).sum().item() / (self.num_labels - 1)
        
        return score
    
    def score_batch(self, paper_texts: List[str], problem_texts: List[str]) -> List[float]:
        """
        Score a batch of paper-problem pairs.
        
        Args:
            paper_texts: List of paper texts
            problem_texts: List of problem texts
            
        Returns:
            list: Relevance scores
        """
        if len(paper_texts) != len(problem_texts):
            raise ValueError("Number of papers and problems must match")
        
        # Prepare inputs
        features = self.tokenizer(
            paper_texts, 
            problem_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Score
        with torch.no_grad():
            outputs = self.model(**features)
            
            # Handle regression or classification models
            if self.num_labels == 1:  # Regression
                scores = outputs.logits.squeeze().cpu().numpy()
                # Normalize to 0-1 range if needed (model dependent)
                scores = (scores + 1) / 2  # STS models output -1 to 1
            else:  # Classification
                probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
                
                # For binary classification, use positive class probability
                # For multi-class, use weighted average
                if self.num_labels == 2:
                    scores = probs[:, 1]
                else:
                    # Assuming the labels represent scores (e.g., 0-5)
                    score_weights = np.arange(self.num_labels) / (self.num_labels - 1)
                    scores = np.sum(probs * score_weights, axis=1)
        
        return scores.tolist()
    
    def rerank_matches(self, paper_id: Any, problem_matches: List[Dict], 
                      batch_size: int = 16) -> List[Dict]:
        """
        Rerank problem matches for a paper using cross-encoder.
        
        Args:
            paper_id: MongoDB ID of the paper
            problem_matches: List of problem matches (from bi-encoder)
            batch_size: Batch size for scoring
            
        Returns:
            list: Reranked problem matches
        """
        if not problem_matches:
            return []
        
        # Get paper from MongoDB
        paper_collection = self.mongo.get_papers_collection()
        paper = paper_collection.find_one({"_id": paper_id})
        
        if not paper:
            logger.error(f"Paper not found: {paper_id}")
            return problem_matches
        
        # Prepare paper text
        paper_title = paper.get("title", "")
        paper_abstract = paper.get("abstract", "")
        paper_text = f"{paper_title} {paper_abstract}".strip()
        
        if not paper_text:
            logger.error(f"Paper {paper_id} has no text content")
            return problem_matches
        
        # Prepare batches
        problem_texts = []
        for match in problem_matches:
            problem_texts.append(match.get("text", ""))
        
        # Score in batches
        all_scores = []
        for i in range(0, len(problem_texts), batch_size):
            batch_problems = problem_texts[i:i + batch_size]
            batch_papers = [paper_text] * len(batch_problems)
            
            batch_scores = self.score_batch(batch_papers, batch_problems)
            all_scores.extend(batch_scores)
        
        # Update scores and rerank
        for i, match in enumerate(problem_matches):
            match["bi_encoder_score"] = match.get("similarity_score", 0)
            match["cross_encoder_score"] = all_scores[i]
            
            # Combined score (you can adjust the weights)
            bi_weight = 0.3
            cross_weight = 0.7
            match["similarity_score"] = (
                bi_weight * match["bi_encoder_score"] + 
                cross_weight * match["cross_encoder_score"]
            )
        
        # Sort by combined score
        reranked_matches = sorted(
            problem_matches, 
            key=lambda x: x["similarity_score"], 
            reverse=True
        )
        
        # Filter by threshold
        reranked_matches = [
            match for match in reranked_matches
            if match["similarity_score"] >= SIMILARITY_THRESHOLD
        ]
        
        return reranked_matches
    
    def rerank_papers(self, problem_id: Any, paper_matches: List[Dict],
                     batch_size: int = 16) -> List[Dict]:
        """
        Rerank paper matches for a problem using cross-encoder.
        
        Args:
            problem_id: MongoDB ID of the problem
            paper_matches: List of paper matches (from bi-encoder)
            batch_size: Batch size for scoring
            
        Returns:
            list: Reranked paper matches
        """
        if not paper_matches:
            return []
        
        # Get problem from MongoDB
        problem_collection = self.mongo.get_problems_collection()
        problem = problem_collection.find_one({"_id": problem_id})
        
        if not problem:
            logger.error(f"Problem not found: {problem_id}")
            return paper_matches
        
        # Prepare problem text
        problem_text = problem.get("text", "")
        
        if not problem_text:
            logger.error(f"Problem {problem_id} has no text content")
            return paper_matches
        
        # Prepare batches
        paper_texts = []
        for match in paper_matches:
            title = match.get("title", "")
            abstract = match.get("abstract", "")
            paper_texts.append(f"{title} {abstract}".strip())
        
        # Score in batches
        all_scores = []
        for i in range(0, len(paper_texts), batch_size):
            batch_papers = paper_texts[i:i + batch_size]
            batch_problems = [problem_text] * len(batch_papers)
            
            batch_scores = self.score_batch(batch_papers, batch_problems)
            all_scores.extend(batch_scores)
        
        # Update scores and rerank
        for i, match in enumerate(paper_matches):
            match["bi_encoder_score"] = match.get("similarity_score", 0)
            match["cross_encoder_score"] = all_scores[i]
            
            # Combined score (adjust weights as needed)
            bi_weight = 0.3
            cross_weight = 0.7
            match["similarity_score"] = (
                bi_weight * match["bi_encoder_score"] + 
                cross_weight * match["cross_encoder_score"]
            )
        
        # Sort by combined score and commercial value
        reranked_matches = sorted(
            paper_matches, 
            key=lambda x: (x["similarity_score"], x.get("commercial_score", 0)),
            reverse=True
        )
        
        # Filter by threshold
        reranked_matches = [
            match for match in reranked_matches
            if match["similarity_score"] >= SIMILARITY_THRESHOLD
        ]
        
        return reranked_matches


def main():
    """Test the cross-encoder reranking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cross-encoder reranking")
    parser.add_argument("--paper-id", type=str, help="MongoDB ID of the paper")
    parser.add_argument("--problem-id", type=str, help="MongoDB ID of the problem")
    
    args = parser.parse_args()
    
    cross_encoder = CrossEncoder()
    
    # Test with example inputs if no IDs provided
    if not args.paper_id and not args.problem_id:
        paper_text = "This paper introduces a novel approach to database optimization for high-throughput applications."
        problem_text = "Our database is slow with many concurrent users. How can we improve performance?"
        
        score = cross_encoder.score_pair(paper_text, problem_text)
        print(f"Relevance score: {score:.4f}")
    
    # Test with real paper and problem
    elif args.paper_id:
        # Get matches from MongoDB
        mongo = get_mongo_connector()
        mongo.connect()
        
        paper_collection = mongo.get_papers_collection()
        paper = paper_collection.find_one({"_id": args.paper_id})
        
        if paper and "matching_problems" in paper:
            matches = paper["matching_problems"]
            print(f"Found {len(matches)} existing matches")
            
            # Rerank
            reranked = cross_encoder.rerank_matches(args.paper_id, matches)
            
            # Print results
            print("\nTop 5 reranked matches:")
            for i, match in enumerate(reranked[:5]):
                print(f"{i+1}. Score: {match['similarity_score']:.4f} (bi: {match.get('bi_encoder_score', 0):.4f}, cross: {match.get('cross_encoder_score', 0):.4f})")
                print(f"   Problem: {match.get('text', '')[:100]}...\n")
    
    elif args.problem_id:
        # Use similarity model to get matches first
        from models.matching.similarity_model import SimilarityModel
        
        similarity_model = SimilarityModel()
        matches = similarity_model.find_similar_papers(args.problem_id)
        
        print(f"Found {len(matches)} matches from bi-encoder")
        
        # Rerank
        reranked = cross_encoder.rerank_papers(args.problem_id, matches)
        
        # Print results
        print("\nTop 5 reranked matches:")
        for i, match in enumerate(reranked[:5]):
            print(f"{i+1}. Score: {match['similarity_score']:.4f} (bi: {match.get('bi_encoder_score', 0):.4f}, cross: {match.get('cross_encoder_score', 0):.4f})")
            print(f"   Paper: {match.get('title', '')}")
            print(f"   Commercial score: {match.get('commercial_score', 0):.4f}\n")


if __name__ == "__main__":
    main()
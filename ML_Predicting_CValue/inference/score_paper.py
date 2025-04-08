#!/usr/bin/env python3
"""
Paper Scoring API function.
Scores a research paper's commercial value and finds related business problems.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_config import CV_MODEL_PATH, TOP_K_PROBLEMS, SIMILARITY_THRESHOLD
from models.commercial_value.cv_model import CommercialValueModel
from models.commercial_value.feature_extraction import CommercialValueFeatureExtractor
from models.matching.similarity_model import SimilarityModel
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperScorer:
    """
    Class to score papers and find related business problems.
    """
    
    def __init__(self, model_path: str = CV_MODEL_PATH):
        """
        Initialize paper scorer.
        
        Args:
            model_path: Path to the commercial value model
        """
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        # Load commercial value model
        self.cv_model = CommercialValueModel(model_path=model_path)
        self.cv_model.load()
        
        # Initialize feature extractor
        self.feature_extractor = CommercialValueFeatureExtractor()
        
        # Initialize similarity model
        self.similarity_model = SimilarityModel()
    
    def score_paper_by_id(self, paper_id: Any) -> Dict:
        """
        Score a paper by its MongoDB ID.
        
        Args:
            paper_id: MongoDB ID of the paper
            
        Returns:
            dict: Scoring results
        """
        # Get paper from MongoDB
        paper_collection = self.mongo.get_papers_collection()
        paper = paper_collection.find_one({"_id": paper_id})
        
        if not paper:
            logger.error(f"Paper not found: {paper_id}")
            return {"error": "Paper not found"}
        
        # Check if paper already has a score
        if "commercial_score" in paper and "matching_problems" in paper:
            # Return existing scores and matches
            result = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "commercial_score": paper.get("commercial_score", 0),
                "score_components": paper.get("score_components", {}),
                "matching_problems": paper.get("matching_problems", [])
            }
            
            logger.info(f"Retrieved existing score for paper {paper_id}")
            return result
        
        # Extract features
        try:
            features = self.feature_extractor.extract_all_features(paper)
            
            # Create feature dataframe
            feature_df = pd.DataFrame([features])
            
            # Predict commercial value
            commercial_score = float(self.cv_model.predict_proba(feature_df)[0])
            commercial_label = 1 if commercial_score >= 0.5 else 0
            
            # Find matching problems
            matching_problems = self.similarity_model.find_similar_problems(
                paper_id,
                k=TOP_K_PROBLEMS,
                threshold=SIMILARITY_THRESHOLD
            )
            
            # Calculate score components (simplified version)
            score_components = {
                "patent_citations": features.get("patent_score", 0),
                "industry_mentions": features.get("industry_score", 0),
                "problem_similarity": features.get("similarity_score", 0),
                "author_affiliation": features.get("affiliation_score", 0)
            }
            
            # Update paper in MongoDB
            paper_collection.update_one(
                {"_id": paper_id},
                {"$set": {
                    "commercial_score": commercial_score,
                    "commercial_label": commercial_label,
                    "score_components": score_components,
                    "matching_problems": matching_problems,
                    "last_scored": datetime.now().isoformat()
                }}
            )
            
            # Return results
            result = {
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "commercial_score": commercial_score,
                "commercial_label": commercial_label,
                "score_components": score_components,
                "matching_problems": matching_problems
            }
            
            logger.info(f"Scored paper {paper_id}: {commercial_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error scoring paper {paper_id}: {e}")
            return {"error": str(e)}
    
    def score_paper_by_text(self, title: str, abstract: str, authors: List[Dict] = None,
                          venue: Dict = None, year: int = None) -> Dict:
        """
        Score a paper by its text content (without storing in DB).
        
        Args:
            title: Paper title
            abstract: Paper abstract
            authors: List of author information (optional)
            venue: Publication venue information (optional)
            year: Publication year (optional)
            
        Returns:
            dict: Scoring results
        """
        # Create paper document
        paper = {
            "title": title,
            "abstract": abstract,
            "authors": authors or [],
            "venue": venue or {},
            "year": year or datetime.now().year
        }
        
        # Extract features
        try:
            features = self.feature_extractor.extract_basic_features(paper)
            
            # Add default values for missing features
            all_features = self.feature_extractor.extract_all_features(paper)
            
            # Create feature dataframe
            feature_df = pd.DataFrame([all_features])
            
            # Predict commercial value
            commercial_score = float(self.cv_model.predict_proba(feature_df)[0])
            commercial_label = 1 if commercial_score >= 0.5 else 0
            
            # Generate embedding for the paper
            from transformers import AutoTokenizer, AutoModel
            import torch
            from config.model_config import SCIBERT_MODEL, MAX_SEQ_LENGTH
            
            tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL)
            model = AutoModel.from_pretrained(SCIBERT_MODEL)
            model.eval()
            
            # Combine title and abstract
            combined_text = f"{title} {abstract}".strip()
            
            # Generate embedding
            tokens = tokenizer(
                combined_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH
            )
            
            with torch.no_grad():
                outputs = model(**tokens)
                cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
            
            # Find similar problems using the embedding
            vector_db = get_vector_connector()
            vector_db.load_problem_index()
            
            similar_problems = vector_db.find_similar_problems(cls_embedding, k=TOP_K_PROBLEMS+10)
            
            # Filter by threshold and get problem details
            similar_problems = [
                (problem_id, score) for problem_id, score in similar_problems
                if score >= SIMILARITY_THRESHOLD
            ][:TOP_K_PROBLEMS]
            
            # Get problem details
            problem_collection = self.mongo.get_problems_collection()
            matching_problems = []
            
            for problem_id, similarity_score in similar_problems:
                problem = problem_collection.find_one({"_id": problem_id})
                if problem:
                    matching_problems.append({
                        "problem_id": problem_id,
                        "text": problem.get("text", ""),
                        "source": problem.get("source", ""),
                        "created_date": problem.get("created_date", ""),
                        "similarity_score": similarity_score,
                        "url": problem.get("url", "")
                    })
            
            # Calculate score components
            score_components = {
                "title_commercial_keywords": features.get("title_commercial_keywords", 0) / 10.0,
                "abstract_commercial_keywords": features.get("abstract_commercial_keywords", 0) / 20.0,
                "problem_similarity": min(1.0, len(matching_problems) / 5.0)
            }
            
            # Return results
            result = {
                "title": title,
                "commercial_score": commercial_score,
                "commercial_label": commercial_label,
                "score_components": score_components,
                "matching_problems": matching_problems
            }
            
            logger.info(f"Scored paper '{title}': {commercial_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error scoring paper text: {e}")
            return {"error": str(e)}


def score_paper_api(paper_id=None, paper_data=None):
    """
    API function to score a paper.
    
    Args:
        paper_id: MongoDB ID of the paper (for existing papers)
        paper_data: Paper text data (for new papers)
        
    Returns:
        dict: Scoring results
    """
    scorer = PaperScorer()
    
    if paper_id:
        return scorer.score_paper_by_id(paper_id)
    elif paper_data:
        return scorer.score_paper_by_text(
            title=paper_data.get("title", ""),
            abstract=paper_data.get("abstract", ""),
            authors=paper_data.get("authors"),
            venue=paper_data.get("venue"),
            year=paper_data.get("year")
        )
    else:
        return {"error": "Missing paper_id or paper_data"}


def main():
    """Test the paper scoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Score a research paper")
    parser.add_argument("--id", type=str, help="MongoDB ID of the paper")
    parser.add_argument("--title", type=str, help="Paper title")
    parser.add_argument("--abstract", type=str, help="Paper abstract")
    parser.add_argument("--year", type=int, help="Publication year")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.id:
        result = score_paper_api(paper_id=args.id)
    elif args.title and args.abstract:
        paper_data = {
            "title": args.title,
            "abstract": args.abstract,
            "year": args.year
        }
        result = score_paper_api(paper_data=paper_data)
    else:
        print("Error: Either provide --id or both --title and --abstract")
        sys.exit(1)
        
    # Print results
    print(json.dumps(result, indent=2))
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
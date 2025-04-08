#!/usr/bin/env python3
"""
Proxy Label Generator for Commercial Value Prediction.
Generates weak labels for papers based on proxy signals like
patent citations, industry mentions, and problem similarity.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import (
    MIN_PATENT_CITATIONS, MIN_SIMILARITY_SCORE, MIN_PROBLEM_MATCHES,
    USE_PATENT_CITATIONS, USE_INDUSTRY_MENTIONS, USE_PROBLEM_SIMILARITY,
    USE_AUTHOR_AFFILIATION
)
from data.connectors.mongo_connector import get_connector as get_mongo_connector
from data.connectors.vector_db_connector import get_connector as get_vector_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProxyLabelGenerator:
    """
    Class to generate weak labels for commercial value based on proxy signals.
    """
    
    def __init__(self):
        self.mongo = get_mongo_connector()
        self.vector_db = get_vector_connector()
        
        # Connect to databases
        if not self.mongo.connect():
            logger.error("Failed to connect to MongoDB")
            raise ConnectionError("Could not connect to MongoDB")
            
        # Load vector indices
        if USE_PROBLEM_SIMILARITY:
            self.vector_db.load_paper_index()
            self.vector_db.load_problem_index()
    
    def get_patent_citation_score(self, paper: Dict) -> float:
        """
        Calculate score based on patent citations.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not USE_PATENT_CITATIONS:
            return 0.0
            
        citations = paper.get("patent_citations", [])
        citation_count = len(citations) if isinstance(citations, list) else 0
        
        # Normalize score: 0 for no citations, 1.0 for meeting/exceeding threshold
        if citation_count >= MIN_PATENT_CITATIONS:
            return 1.0
        elif citation_count > 0:
            return citation_count / MIN_PATENT_CITATIONS
        else:
            return 0.0
    
    def get_industry_mention_score(self, paper: Dict) -> float:
        """
        Calculate score based on industry mentions.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not USE_INDUSTRY_MENTIONS:
            return 0.0
            
        # Get industry mentions from various sources
        mentions = paper.get("industry_mentions", {})
        
        # Calculate mentions from different sources
        news_mentions = len(mentions.get("news", [])) if isinstance(mentions.get("news"), list) else 0
        blog_mentions = len(mentions.get("blogs", [])) if isinstance(mentions.get("blogs"), list) else 0
        company_mentions = len(mentions.get("companies", [])) if isinstance(mentions.get("companies"), list) else 0
        
        # Weight different sources
        weighted_score = (
            news_mentions * 0.3 +
            blog_mentions * 0.4 +
            company_mentions * 0.5
        )
        
        # Normalize score: cap at 1.0
        return min(1.0, weighted_score / 3.0)
    
    def get_problem_similarity_score(self, paper: Dict) -> Tuple[float, List[Dict]]:
        """
        Calculate score based on similarity to business problems.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            tuple: (score, list of matching problems)
        """
        if not USE_PROBLEM_SIMILARITY:
            return 0.0, []
            
        # Check if we have embedding
        embedding = paper.get("scibert_embedding")
        if not embedding:
            logger.warning(f"Paper {paper['_id']} has no embedding, skipping similarity check")
            return 0.0, []
            
        # Find similar problems
        similar_problems = self.vector_db.find_similar_problems(
            np.array(embedding),
            k=20  # Get more than needed for filtering
        )
        
        # Filter problems with high similarity
        high_similarity_problems = [
            (problem_id, score) for problem_id, score in similar_problems
            if score >= MIN_SIMILARITY_SCORE
        ]
        
        # Get problem details
        problem_collection = self.mongo.get_problems_collection()
        matching_problems = []
        
        for problem_id, score in high_similarity_problems:
            problem_doc = problem_collection.find_one({"_id": problem_id})
            if problem_doc:
                matching_problems.append({
                    "problem_id": problem_id,
                    "text": problem_doc.get("text", ""),
                    "source": problem_doc.get("source", ""),
                    "similarity_score": score
                })
        
        # Calculate score based on number of high-similarity matches
        if len(matching_problems) >= MIN_PROBLEM_MATCHES:
            similarity_score = 1.0
        elif len(matching_problems) > 0:
            similarity_score = len(matching_problems) / MIN_PROBLEM_MATCHES
        else:
            similarity_score = 0.0
            
        return similarity_score, matching_problems
    
    def get_author_affiliation_score(self, paper: Dict) -> float:
        """
        Calculate score based on author affiliations (industry vs. academic).
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not USE_AUTHOR_AFFILIATION:
            return 0.0
            
        authors = paper.get("authors", [])
        if not authors:
            return 0.0
            
        # Count industry affiliations
        industry_count = 0
        for author in authors:
            affiliation = author.get("affiliation", "").lower()
            
            # Check for industry keywords in affiliation
            industry_keywords = [
                "inc", "llc", "ltd", "corp", "gmbh", "company", "technologies", 
                "labs", "research center", "r&d", "innovation"
            ]
            
            if any(keyword in affiliation for keyword in industry_keywords):
                industry_count += 1
                
        # Calculate ratio of industry to total authors
        industry_ratio = industry_count / len(authors)
        
        # Apply weightings to get a score
        if industry_ratio >= 0.5:  # More than half from industry
            return 1.0
        elif industry_count > 0:  # Some industry authors
            return 0.5 + (industry_ratio * 0.5)  # Scale 0.5 to 1.0
        else:
            return 0.0
    
    def generate_commercial_score(self, paper: Dict) -> Dict:
        """
        Generate commercial value score and labels for a paper.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Updated paper with commercial_score and related fields
        """
        paper_id = paper["_id"]
        
        # Get individual scores
        patent_score = self.get_patent_citation_score(paper)
        industry_score = self.get_industry_mention_score(paper)
        similarity_score, matching_problems = self.get_problem_similarity_score(paper)
        affiliation_score = self.get_author_affiliation_score(paper)
        
        # Calculate weighted commercial score
        component_weights = {
            "patent_citations": 0.4,
            "industry_mentions": 0.2,
            "problem_similarity": 0.3,
            "author_affiliation": 0.1
        }
        
        commercial_score = (
            patent_score * component_weights["patent_citations"] +
            industry_score * component_weights["industry_mentions"] +
            similarity_score * component_weights["problem_similarity"] +
            affiliation_score * component_weights["author_affiliation"]
        )
        
        # Create binary label: 1 for high commercial potential, 0 for low
        commercial_label = 1 if commercial_score >= 0.5 else 0
        
        # Prepare update
        update_data = {
            "commercial_score": commercial_score,
            "commercial_label": commercial_label,
            "score_components": {
                "patent_citations": patent_score,
                "industry_mentions": industry_score,
                "problem_similarity": similarity_score,
                "author_affiliation": affiliation_score
            }
        }
        
        # Add matching problems if any
        if matching_problems:
            update_data["matching_problems"] = matching_problems
            
        return update_data
    
    def generate_labels_for_all_papers(self, limit: int = 0, save: bool = True) -> List[Dict]:
        """
        Generate commercial value labels for all papers.
        
        Args:
            limit: Maximum number of papers to process (0 for all)
            save: Whether to save results to MongoDB
            
        Returns:
            list: List of processed papers with commercial value scores
        """
        # Get all papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        query = {"scibert_embedding": {"$exists": True}}
        
        cursor = paper_collection.find(query)
        if limit > 0:
            cursor = cursor.limit(limit)
            
        papers = list(cursor)
        logger.info(f"Generating commercial value labels for {len(papers)} papers")
        
        results = []
        
        for paper in tqdm(papers):
            try:
                # Generate score
                update_data = self.generate_commercial_score(paper)
                
                # Store the result
                if save:
                    paper_collection.update_one(
                        {"_id": paper["_id"]},
                        {"$set": update_data}
                    )
                
                # Add to results
                paper_result = {
                    "paper_id": paper["_id"],
                    "title": paper.get("title", ""),
                    "commercial_score": update_data["commercial_score"],
                    "commercial_label": update_data["commercial_label"]
                }
                results.append(paper_result)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('_id')}: {e}")
        
        logger.info(f"Completed generating labels for {len(results)} papers")
        return results


def main():
    """Main function to generate proxy labels for all papers."""
    generator = ProxyLabelGenerator()
    generator.generate_labels_for_all_papers()
    
    
if __name__ == "__main__":
    main()
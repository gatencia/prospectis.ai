#!/usr/bin/env python3
"""
Related Papers Finder.
Finds research papers related to a business problem.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_config import TOP_K_PAPERS, SIMILARITY_THRESHOLD
from models.matching.similarity_model import SimilarityModel
from data.connectors.mongo_connector import get_connector as get_mongo_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RelatedPapersFinder:
    """
    Class to find research papers related to business problems.
    """
    
    def __init__(self):
        """Initialize the related papers finder."""
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        # Initialize similarity model
        self.similarity_model = SimilarityModel()
    
    def find_papers_for_problem_id(self, problem_id: Any, k: int = TOP_K_PAPERS,
                                 threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        """
        Find papers related to a specific business problem by ID.
        
        Args:
            problem_id: MongoDB ID of the problem
            k: Number of papers to return
            threshold: Minimum similarity threshold
            
        Returns:
            dict: Problem details with related papers
        """
        # Get problem from MongoDB
        problem_collection = self.mongo.get_problems_collection()
        problem = problem_collection.find_one({"_id": problem_id})
        
        if not problem:
            logger.error(f"Problem not found: {problem_id}")
            return {"error": "Problem not found"}
        
        # Find similar papers
        similar_papers = self.similarity_model.find_similar_papers(
            problem_id,
            k=k,
            threshold=threshold
        )
        
        # Format response
        result = {
            "problem_id": problem_id,
            "problem_text": problem.get("text", ""),
            "problem_source": problem.get("source", ""),
            "problem_url": problem.get("url", ""),
            "created_date": problem.get("created_date", ""),
            "related_papers": similar_papers,
            "total_papers": len(similar_papers)
        }
        
        logger.info(f"Found {len(similar_papers)} papers for problem {problem_id}")
        return result
    
    def find_papers_for_problem_text(self, problem_text: str, k: int = TOP_K_PAPERS,
                                   threshold: float = SIMILARITY_THRESHOLD) -> Dict:
        """
        Find papers related to a problem described in text.
        
        Args:
            problem_text: Description of the business problem
            k: Number of papers to return
            threshold: Minimum similarity threshold
            
        Returns:
            dict: Problem details with related papers
        """
        # Find similar papers directly from text
        similar_papers = self.similarity_model.find_papers_for_text(
            problem_text,
            k=k,
            threshold=threshold
        )
        
        # Format response
        result = {
            "problem_text": problem_text,
            "related_papers": similar_papers,
            "total_papers": len(similar_papers)
        }
        
        logger.info(f"Found {len(similar_papers)} papers for problem text")
        return result
    
    def find_papers_by_keyword(self, keyword: str, k: int = TOP_K_PAPERS) -> Dict:
        """
        Find papers related to a keyword using MongoDB text search.
        
        Args:
            keyword: Search keyword
            k: Number of papers to return
            
        Returns:
            dict: Search results with matching papers
        """
        # Search papers by keyword
        paper_collection = self.mongo.get_papers_collection()
        
        # Use MongoDB text search
        cursor = paper_collection.find(
            {"$text": {"$search": keyword}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(k)
        
        papers = list(cursor)
        
        # Format paper details
        paper_results = []
        
        for paper in papers:
            paper_results.append({
                "paper_id": paper["_id"],
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year", ""),
                "venue": paper.get("venue", {}),
                "url": paper.get("url", ""),
                "doi": paper.get("doi", ""),
                "search_score": paper.get("score", 0),
                "commercial_score": paper.get("commercial_score", 0)
            })
            
        # Also search problems to show the business context
        problem_collection = self.mongo.get_problems_collection()
        problem_cursor = problem_collection.find(
            {"$text": {"$search": keyword}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(5)
        
        related_problems = []
        
        for problem in problem_cursor:
            related_problems.append({
                "problem_id": problem["_id"],
                "text": problem.get("text", ""),
                "source": problem.get("source", ""),
                "url": problem.get("url", ""),
                "search_score": problem.get("score", 0)
            })
        
        # Format response
        result = {
            "keyword": keyword,
            "related_papers": paper_results,
            "total_papers": len(paper_results),
            "related_problems": related_problems
        }
        
        logger.info(f"Found {len(paper_results)} papers for keyword '{keyword}'")
        return result


def find_related_papers_api(problem_id=None, problem_text=None, keyword=None, k=TOP_K_PAPERS):
    """
    API function to find papers related to a business problem.
    
    Args:
        problem_id: MongoDB ID of the problem
        problem_text: Problem description text
        keyword: Search keyword
        k: Number of papers to return
        
    Returns:
        dict: Related papers
    """
    finder = RelatedPapersFinder()
    
    if problem_id:
        return finder.find_papers_for_problem_id(problem_id, k=k)
    elif problem_text:
        return finder.find_papers_for_problem_text(problem_text, k=k)
    elif keyword:
        return finder.find_papers_by_keyword(keyword, k=k)
    else:
        return {"error": "Missing problem_id, problem_text, or keyword"}


def main():
    """Test the related papers finder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find research papers related to a business problem")
    parser.add_argument("--id", type=str, help="MongoDB ID of the problem")
    parser.add_argument("--text", type=str, help="Problem description text")
    parser.add_argument("--keyword", type=str, help="Search keyword")
    parser.add_argument("--k", type=int, default=TOP_K_PAPERS, help="Number of papers to return")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if args.id:
        result = find_related_papers_api(problem_id=args.id, k=args.k)
    elif args.text:
        result = find_related_papers_api(problem_text=args.text, k=args.k)
    elif args.keyword:
        result = find_related_papers_api(keyword=args.keyword, k=args.k)
    else:
        print("Error: Provide one of --id, --text, or --keyword")
        sys.exit(1)
        
    # Print results
    print(json.dumps(result, indent=2))
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
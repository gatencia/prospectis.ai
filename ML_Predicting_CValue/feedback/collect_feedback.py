#!/usr/bin/env python3
"""
Collect feedback on paper-problem matches and commercial value predictions.
Provides functions to store and validate user feedback.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.connectors.mongo_connector import get_connector as get_mongo_connector
from utils.logging import setup_logger

# Configure logging
logger = setup_logger("feedback_collector", console=True)


class FeedbackCollector:
    """
    Collect and store user feedback on the Prospectis system.
    """
    
    def __init__(self):
        """Initialize feedback collector."""
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        # Get collections
        self.feedback_collection = self.mongo.get_collection("user_feedback")
        self.papers_collection = self.mongo.get_papers_collection()
        self.problems_collection = self.mongo.get_problems_collection()
        
        if not self.feedback_collection:
            logger.error("Failed to get feedback collection")
            raise ConnectionError("Could not connect to feedback collection")
    
    def validate_ids(self, paper_id: Optional[Any] = None, 
                   problem_id: Optional[Any] = None) -> bool:
        """
        Validate paper and problem IDs exist in the database.
        
        Args:
            paper_id: Paper ID to validate
            problem_id: Problem ID to validate
            
        Returns:
            bool: True if IDs are valid
        """
        # Validate paper ID if provided
        if paper_id is not None:
            paper = self.papers_collection.find_one({"_id": paper_id})
            if not paper:
                logger.warning(f"Paper ID not found: {paper_id}")
                return False
        
        # Validate problem ID if provided
        if problem_id is not None:
            problem = self.problems_collection.find_one({"_id": problem_id})
            if not problem:
                logger.warning(f"Problem ID not found: {problem_id}")
                return False
        
        return True
    
    def add_paper_feedback(self, paper_id: Any, user_id: Optional[str] = None,
                         rating: Optional[int] = None, 
                         feedback_text: Optional[str] = None,
                         feedback_type: Optional[str] = None) -> Dict:
        """
        Add feedback for a paper's commercial value.
        
        Args:
            paper_id: MongoDB ID of the paper
            user_id: Optional user identifier
            rating: Rating (typically 1-5)
            feedback_text: Optional feedback text
            feedback_type: Type of feedback (e.g., "relevance", "accuracy")
            
        Returns:
            dict: Operation result
        """
        # Validate paper ID
        if not self.validate_ids(paper_id=paper_id):
            return {"error": "Invalid paper ID"}
        
        # Validate rating if provided
        if rating is not None and (not isinstance(rating, int) or rating < 1 or rating > 5):
            return {"error": "Rating must be an integer between 1 and 5"}
        
        # Create feedback entry
        feedback_entry = {
            "paper_id": paper_id,
            "timestamp": datetime.now(),
            "feedback_source": "api"
        }
        
        # Add optional fields if provided
        if user_id is not None:
            feedback_entry["user_id"] = user_id
        
        if rating is not None:
            feedback_entry["rating"] = rating
        
        if feedback_text is not None:
            feedback_entry["feedback_text"] = feedback_text
        
        if feedback_type is not None:
            feedback_entry["feedback_type"] = feedback_type
        
        # Store feedback
        result = self.feedback_collection.insert_one(feedback_entry)
        
        if not result.inserted_id:
            logger.error("Failed to insert feedback")
            return {"error": "Failed to store feedback"}
        
        logger.info(f"Added paper feedback for {paper_id} (feedback ID: {result.inserted_id})")
        
        return {
            "success": True,
            "feedback_id": result.inserted_id,
            "timestamp": feedback_entry["timestamp"].isoformat()
        }
    
    def add_match_feedback(self, paper_id: Any, problem_id: Any, 
                         user_id: Optional[str] = None,
                         match_rating: Optional[int] = None,
                         feedback_text: Optional[str] = None,
                         relevance_type: Optional[str] = None) -> Dict:
        """
        Add feedback for a paper-problem match.
        
        Args:
            paper_id: MongoDB ID of the paper
            problem_id: MongoDB ID of the problem
            user_id: Optional user identifier
            match_rating: Rating (typically 1-5)
            feedback_text: Optional feedback text
            relevance_type: Type of relevance (e.g., "direct", "partial")
            
        Returns:
            dict: Operation result
        """
        # Validate IDs
        if not self.validate_ids(paper_id=paper_id, problem_id=problem_id):
            return {"error": "Invalid paper or problem ID"}
        
        # Validate rating if provided
        if match_rating is not None and (not isinstance(match_rating, int) or match_rating < 1 or match_rating > 5):
            return {"error": "Match rating must be an integer between 1 and 5"}
        
        # Create feedback entry
        feedback_entry = {
            "paper_id": paper_id,
            "problem_id": problem_id,
            "timestamp": datetime.now(),
            "feedback_source": "api"
        }
        
        # Add optional fields if provided
        if user_id is not None:
            feedback_entry["user_id"] = user_id
        
        if match_rating is not None:
            feedback_entry["match_rating"] = match_rating
        
        if feedback_text is not None:
            feedback_entry["feedback_text"] = feedback_text
        
        if relevance_type is not None:
            feedback_entry["relevance_type"] = relevance_type
        
        # Store feedback
        result = self.feedback_collection.insert_one(feedback_entry)
        
        if not result.inserted_id:
            logger.error("Failed to insert match feedback")
            return {"error": "Failed to store feedback"}
        
        logger.info(f"Added match feedback for paper {paper_id} and problem {problem_id} (feedback ID: {result.inserted_id})")
        
        return {
            "success": True,
            "feedback_id": result.inserted_id,
            "timestamp": feedback_entry["timestamp"].isoformat()
        }
    
    def add_search_feedback(self, query_text: str, paper_ids: List[Any],
                          clicked_paper_id: Optional[Any] = None,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> Dict:
        """
        Add feedback for search results.
        
        Args:
            query_text: Search query text
            paper_ids: List of paper IDs returned in search
            clicked_paper_id: ID of paper that was clicked (if any)
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            dict: Operation result
        """
        # Validate clicked paper ID if provided
        if clicked_paper_id is not None and not self.validate_ids(paper_id=clicked_paper_id):
            return {"error": "Invalid clicked paper ID"}
        
        # Create feedback entry
        feedback_entry = {
            "query_text": query_text,
            "paper_ids": paper_ids,
            "result_count": len(paper_ids),
            "timestamp": datetime.now(),
            "feedback_type": "search",
            "feedback_source": "api"
        }
        
        # Add optional fields if provided
        if clicked_paper_id is not None:
            feedback_entry["clicked_paper_id"] = clicked_paper_id
        
        if user_id is not None:
            feedback_entry["user_id"] = user_id
        
        if session_id is not None:
            feedback_entry["session_id"] = session_id
        
        # Store feedback
        result = self.feedback_collection.insert_one(feedback_entry)
        
        if not result.inserted_id:
            logger.error("Failed to insert search feedback")
            return {"error": "Failed to store feedback"}
        
        logger.info(f"Added search feedback for query '{query_text}' (feedback ID: {result.inserted_id})")
        
        return {
            "success": True,
            "feedback_id": result.inserted_id,
            "timestamp": feedback_entry["timestamp"].isoformat()
        }
    
    def collect_bulk_feedback(self, feedback_file: Union[str, Path]) -> Dict:
        """
        Process a bulk feedback file.
        
        Args:
            feedback_file: Path to JSON file with feedback
            
        Returns:
            dict: Processing results
        """
        try:
            # Load feedback file
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
            
            if not isinstance(feedback_data, list):
                return {"error": "Feedback file must contain a list of feedback entries"}
            
            logger.info(f"Loaded {len(feedback_data)} feedback entries from {feedback_file}")
            
            # Process each entry
            paper_feedback_count = 0
            match_feedback_count = 0
            search_feedback_count = 0
            failed_entries = []
            
            for i, entry in enumerate(feedback_data):
                try:
                    entry_type = entry.get("type", "")
                    
                    if entry_type == "paper":
                        result = self.add_paper_feedback(
                            paper_id=entry.get("paper_id"),
                            user_id=entry.get("user_id"),
                            rating=entry.get("rating"),
                            feedback_text=entry.get("feedback_text"),
                            feedback_type=entry.get("feedback_type")
                        )
                        
                        if result.get("success"):
                            paper_feedback_count += 1
                        else:
                            failed_entries.append({"index": i, "error": result.get("error")})
                            
                    elif entry_type == "match":
                        result = self.add_match_feedback(
                            paper_id=entry.get("paper_id"),
                            problem_id=entry.get("problem_id"),
                            user_id=entry.get("user_id"),
                            match_rating=entry.get("match_rating"),
                            feedback_text=entry.get("feedback_text"),
                            relevance_type=entry.get("relevance_type")
                        )
                        
                        if result.get("success"):
                            match_feedback_count += 1
                        else:
                            failed_entries.append({"index": i, "error": result.get("error")})
                            
                    elif entry_type == "search":
                        result = self.add_search_feedback(
                            query_text=entry.get("query_text"),
                            paper_ids=entry.get("paper_ids", []),
                            clicked_paper_id=entry.get("clicked_paper_id"),
                            user_id=entry.get("user_id"),
                            session_id=entry.get("session_id")
                        )
                        
                        if result.get("success"):
                            search_feedback_count += 1
                        else:
                            failed_entries.append({"index": i, "error": result.get("error")})
                            
                    else:
                        failed_entries.append({"index": i, "error": f"Unknown feedback type: {entry_type}"})
                        
                except Exception as e:
                    failed_entries.append({"index": i, "error": str(e)})
            
            # Return results
            return {
                "success": True,
                "paper_feedback_count": paper_feedback_count,
                "match_feedback_count": match_feedback_count,
                "search_feedback_count": search_feedback_count,
                "total_processed": paper_feedback_count + match_feedback_count + search_feedback_count,
                "failed_entries": failed_entries
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback file: {e}")
            return {"error": f"Failed to process feedback file: {str(e)}"}
    
    def get_feedback_for_paper(self, paper_id: Any) -> List[Dict]:
        """
        Get all feedback for a specific paper.
        
        Args:
            paper_id: MongoDB ID of the paper
            
        Returns:
            list: Feedback entries
        """
        cursor = self.feedback_collection.find({
            "paper_id": paper_id
        }).sort("timestamp", -1)
        
        feedback_list = list(cursor)
        
        # Convert ObjectId to string and datetime to string
        for item in feedback_list:
            item["_id"] = str(item["_id"])
            if "timestamp" in item:
                item["timestamp"] = item["timestamp"].isoformat()
        
        return feedback_list
    
    def get_feedback_for_match(self, paper_id: Any, problem_id: Any) -> List[Dict]:
        """
        Get all feedback for a specific paper-problem match.
        
        Args:
            paper_id: MongoDB ID of the paper
            problem_id: MongoDB ID of the problem
            
        Returns:
            list: Feedback entries
        """
        cursor = self.feedback_collection.find({
            "paper_id": paper_id,
            "problem_id": problem_id
        }).sort("timestamp", -1)
        
        feedback_list = list(cursor)
        
        # Convert ObjectId to string and datetime to string
        for item in feedback_list:
            item["_id"] = str(item["_id"])
            if "timestamp" in item:
                item["timestamp"] = item["timestamp"].isoformat()
        
        return feedback_list


def feedback_api(action: str, data: Dict) -> Dict:
    """
    API function for feedback collection.
    
    Args:
        action: Action to perform ("paper", "match", "search", "bulk")
        data: Data for the action
        
    Returns:
        dict: Operation result
    """
    collector = FeedbackCollector()
    
    if action == "paper":
        return collector.add_paper_feedback(
            paper_id=data.get("paper_id"),
            user_id=data.get("user_id"),
            rating=data.get("rating"),
            feedback_text=data.get("feedback_text"),
            feedback_type=data.get("feedback_type")
        )
    
    elif action == "match":
        return collector.add_match_feedback(
            paper_id=data.get("paper_id"),
            problem_id=data.get("problem_id"),
            user_id=data.get("user_id"),
            match_rating=data.get("match_rating"),
            feedback_text=data.get("feedback_text"),
            relevance_type=data.get("relevance_type")
        )
    
    elif action == "search":
        return collector.add_search_feedback(
            query_text=data.get("query_text"),
            paper_ids=data.get("paper_ids", []),
            clicked_paper_id=data.get("clicked_paper_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id")
        )
    
    elif action == "bulk":
        feedback_file = data.get("feedback_file")
        if not feedback_file:
            return {"error": "Missing feedback_file parameter"}
        
        return collector.collect_bulk_feedback(feedback_file)
    
    else:
        return {"error": f"Unknown action: {action}"}


def main():
    """Command-line interface for feedback collection."""
    parser = argparse.ArgumentParser(description="Collect feedback for Prospectis ML")
    
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Paper feedback parser
    paper_parser = subparsers.add_parser("paper", help="Add paper feedback")
    paper_parser.add_argument("--paper-id", required=True, help="MongoDB ID of the paper")
    paper_parser.add_argument("--user-id", help="User identifier")
    paper_parser.add_argument("--rating", type=int, help="Rating (1-5)")
    paper_parser.add_argument("--feedback-text", help="Feedback text")
    paper_parser.add_argument("--feedback-type", help="Type of feedback")
    
    # Match feedback parser
    match_parser = subparsers.add_parser("match", help="Add match feedback")
    match_parser.add_argument("--paper-id", required=True, help="MongoDB ID of the paper")
    match_parser.add_argument("--problem-id", required=True, help="MongoDB ID of the problem")
    match_parser.add_argument("--user-id", help="User identifier")
    match_parser.add_argument("--match-rating", type=int, help="Match rating (1-5)")
    match_parser.add_argument("--feedback-text", help="Feedback text")
    match_parser.add_argument("--relevance-type", help="Type of relevance")
    
    # Search feedback parser
    search_parser = subparsers.add_parser("search", help="Add search feedback")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--paper-ids", required=True, help="Comma-separated list of paper IDs")
    search_parser.add_argument("--clicked-paper-id", help="ID of clicked paper")
    search_parser.add_argument("--user-id", help="User identifier")
    search_parser.add_argument("--session-id", help="Session identifier")
    
    # Bulk feedback parser
    bulk_parser = subparsers.add_parser("bulk", help="Process bulk feedback file")
    bulk_parser.add_argument("--file", required=True, help="Path to JSON file with feedback")
    
    # Get feedback parser
    get_parser = subparsers.add_parser("get", help="Get feedback")
    get_parser.add_argument("--paper-id", required=True, help="MongoDB ID of the paper")
    get_parser.add_argument("--problem-id", help="MongoDB ID of the problem")
    get_parser.add_argument("--output", help="Path to save output JSON")
    
    args = parser.parse_args()
    
    # Handle actions
    collector = FeedbackCollector()
    
    if args.action == "paper":
        result = collector.add_paper_feedback(
            paper_id=args.paper_id,
            user_id=args.user_id,
            rating=args.rating,
            feedback_text=args.feedback_text,
            feedback_type=args.feedback_type
        )
        print(json.dumps(result, indent=2))
    
    elif args.action == "match":
        result = collector.add_match_feedback(
            paper_id=args.paper_id,
            problem_id=args.problem_id,
            user_id=args.user_id,
            match_rating=args.match_rating,
            feedback_text=args.feedback_text,
            relevance_type=args.relevance_type
        )
        print(json.dumps(result, indent=2))
    
    elif args.action == "search":
        paper_ids = args.paper_ids.split(",")
        result = collector.add_search_feedback(
            query_text=args.query,
            paper_ids=paper_ids,
            clicked_paper_id=args.clicked_paper_id,
            user_id=args.user_id,
            session_id=args.session_id
        )
        print(json.dumps(result, indent=2))
    
    elif args.action == "bulk":
        result = collector.collect_bulk_feedback(args.file)
        print(json.dumps(result, indent=2))
    
    elif args.action == "get":
        if args.problem_id:
            feedback = collector.get_feedback_for_match(args.paper_id, args.problem_id)
            print(f"Found {len(feedback)} feedback entries for paper-problem match")
        else:
            feedback = collector.get_feedback_for_paper(args.paper_id)
            print(f"Found {len(feedback)} feedback entries for paper")
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(feedback, f, indent=2)
            print(f"Saved feedback to {args.output}")
        else:
            print(json.dumps(feedback, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
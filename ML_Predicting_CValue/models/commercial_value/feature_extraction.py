#!/usr/bin/env python3
"""
Feature Extraction for Commercial Value Prediction.
Extracts features from papers for use in the commercial value prediction model.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import re
from tqdm import tqdm
import pandas as pd

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import (
    USE_PATENT_CITATIONS, USE_INDUSTRY_MENTIONS, USE_PROBLEM_SIMILARITY,
    USE_AUTHOR_AFFILIATION, USE_LLM_SCORING
)
from data.connectors.mongo_connector import get_connector as get_mongo_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommercialValueFeatureExtractor:
    """
    Extract features from papers for commercial value prediction.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        # Keywords indicating commercial potential
        self.commercial_keywords = [
            "application", "industry", "implementation", "product", "commercial",
            "market", "startup", "business", "company", "deploy", "prototype",
            "production", "practical", "real-world", "solution", "customer",
            "use case", "client", "patent", "license", "revenue", "monetize"
        ]
        
        # Keywords indicating theoretical focus
        self.theoretical_keywords = [
            "theorem", "proof", "lemma", "theoretical", "asymptotic", "formalism",
            "abstract", "conjecture", "mathematical", "formal", "concept", "philosophy",
            "constraint", "hypothesis", "axiom", "primitive", "property"
        ]
        
    def extract_basic_features(self, paper: Dict) -> Dict:
        """
        Extract basic paper features (metadata-based).
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        # Paper metadata features
        features["has_abstract"] = 1 if paper.get("abstract") else 0
        features["has_full_text"] = 1 if paper.get("full_text") else 0
        
        # Title features
        title = paper.get("title", "").lower()
        
        # Title length
        features["title_length"] = len(title.split())
        
        # Title contains commercial keywords
        features["title_commercial_keywords"] = sum(1 for kw in self.commercial_keywords if kw in title)
        
        # Title contains theoretical keywords
        features["title_theoretical_keywords"] = sum(1 for kw in self.theoretical_keywords if kw in title)
        
        # Abstract features if available
        abstract = paper.get("abstract", "").lower()
        if abstract:
            # Abstract length
            features["abstract_length"] = len(abstract.split())
            
            # Abstract contains commercial keywords
            features["abstract_commercial_keywords"] = sum(1 for kw in self.commercial_keywords if kw in abstract)
            
            # Abstract contains theoretical keywords
            features["abstract_theoretical_keywords"] = sum(1 for kw in self.theoretical_keywords if kw in abstract)
            
            # Look for phrases indicating application
            application_phrases = ["we apply", "application to", "can be used", "is applied to", 
                                  "practical", "industry", "real-world", "implement"]
            features["abstract_application_phrases"] = sum(1 for phrase in application_phrases if phrase in abstract)
        else:
            features["abstract_length"] = 0
            features["abstract_commercial_keywords"] = 0
            features["abstract_theoretical_keywords"] = 0
            features["abstract_application_phrases"] = 0
        
        # Publication venue features
        venue = paper.get("venue", {})
        venue_name = venue.get("name", "").lower()
        venue_type = venue.get("type", "").lower()
        
        # Is applied venue
        applied_venues = ["applied", "practical", "industry", "implementation"]
        features["is_applied_venue"] = 1 if any(term in venue_name for term in applied_venues) else 0
        
        # Is theoretical venue
        theoretical_venues = ["theory", "theoretical", "foundations", "mathematics"]
        features["is_theoretical_venue"] = 1 if any(term in venue_name for term in theoretical_venues) else 0
        
        # Conference vs journal
        features["is_conference"] = 1 if venue_type == "conference" else 0
        features["is_journal"] = 1 if venue_type == "journal" else 0
        
        # Publication year (normalized)
        current_year = 2025  # Update as needed
        pub_year = paper.get("year", current_year)
        if isinstance(pub_year, int) and pub_year > 1900:
            features["years_since_publication"] = current_year - pub_year
        else:
            features["years_since_publication"] = 0
            
        return features
    
    def extract_citation_features(self, paper: Dict) -> Dict:
        """
        Extract citation-related features.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        if USE_PATENT_CITATIONS:
            # Patent citation features
            patent_citations = paper.get("patent_citations", [])
            features["patent_citation_count"] = len(patent_citations) if isinstance(patent_citations, list) else 0
            
            # Patents by year (if available)
            if isinstance(patent_citations, list) and patent_citations:
                patent_years = [citation.get("year", 0) for citation in patent_citations if isinstance(citation, dict)]
                if patent_years:
                    features["patent_recency"] = max(patent_years) if max(patent_years) > 0 else 0
                    features["patent_years_span"] = max(patent_years) - min(patent_years) if len(patent_years) > 1 else 0
                else:
                    features["patent_recency"] = 0
                    features["patent_years_span"] = 0
            else:
                features["patent_recency"] = 0
                features["patent_years_span"] = 0
        else:
            features["patent_citation_count"] = 0
            features["patent_recency"] = 0
            features["patent_years_span"] = 0
        
        # Academic citation features
        academic_citations = paper.get("citations", [])
        features["academic_citation_count"] = len(academic_citations) if isinstance(academic_citations, list) else 0
        
        # Citation velocity (if available)
        citation_by_year = paper.get("citation_by_year", {})
        if citation_by_year and isinstance(citation_by_year, dict):
            years = sorted([int(year) for year in citation_by_year.keys() if year.isdigit()])
            if len(years) >= 2:
                # Calculate citation growth from first to last year
                first_year = str(years[0])
                last_year = str(years[-1])
                first_count = citation_by_year.get(first_year, 0)
                last_count = citation_by_year.get(last_year, 0)
                features["citation_growth"] = last_count - first_count
                features["citation_velocity"] = features["citation_growth"] / (years[-1] - years[0]) if (years[-1] - years[0]) > 0 else 0
            else:
                features["citation_growth"] = 0
                features["citation_velocity"] = 0
        else:
            features["citation_growth"] = 0
            features["citation_velocity"] = 0
            
        return features
    
    def extract_author_features(self, paper: Dict) -> Dict:
        """
        Extract author-related features.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        if USE_AUTHOR_AFFILIATION:
            authors = paper.get("authors", [])
            
            # Number of authors
            features["author_count"] = len(authors)
            
            # Count industry vs academic affiliations
            industry_count = 0
            academic_count = 0
            
            industry_keywords = [
                "inc", "llc", "ltd", "corp", "gmbh", "company", "technologies", 
                "labs", "research center", "r&d", "innovation"
            ]
            
            academic_keywords = [
                "university", "college", "institute", "school", "academy", 
                "faculty", "department", "universität", "universidad"
            ]
            
            for author in authors:
                affiliation = author.get("affiliation", "").lower()
                
                if any(keyword in affiliation for keyword in industry_keywords):
                    industry_count += 1
                    
                if any(keyword in affiliation for keyword in academic_keywords):
                    academic_count += 1
            
            features["industry_author_count"] = industry_count
            features["academic_author_count"] = academic_count
            
            # Calculate ratios if there are authors
            if features["author_count"] > 0:
                features["industry_author_ratio"] = industry_count / features["author_count"]
                features["academic_author_ratio"] = academic_count / features["author_count"]
            else:
                features["industry_author_ratio"] = 0
                features["academic_author_ratio"] = 0
                
            # Collaboration type
            if industry_count > 0 and academic_count > 0:
                features["is_industry_academic_collaboration"] = 1
            else:
                features["is_industry_academic_collaboration"] = 0
        else:
            features["author_count"] = 0
            features["industry_author_count"] = 0
            features["academic_author_count"] = 0
            features["industry_author_ratio"] = 0
            features["academic_author_ratio"] = 0
            features["is_industry_academic_collaboration"] = 0
            
        return features
    
    def extract_problem_similarity_features(self, paper: Dict) -> Dict:
        """
        Extract features related to business problem similarity.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        if USE_PROBLEM_SIMILARITY:
            # Get matching problems
            matching_problems = paper.get("matching_problems", [])
            
            # Number of matching problems
            features["matching_problem_count"] = len(matching_problems)
            
            # Similarity scores
            similarity_scores = [p.get("similarity_score", 0) for p in matching_problems]
            
            if similarity_scores:
                features["max_problem_similarity"] = max(similarity_scores)
                features["avg_problem_similarity"] = sum(similarity_scores) / len(similarity_scores)
                
                # High similarity problems
                high_similarity = [s for s in similarity_scores if s >= 0.8]
                features["high_similarity_count"] = len(high_similarity)
            else:
                features["max_problem_similarity"] = 0
                features["avg_problem_similarity"] = 0
                features["high_similarity_count"] = 0
                
            # Problem sources
            problem_sources = [p.get("source", "") for p in matching_problems]
            source_counts = Counter(problem_sources)
            
            # Features for popular sources
            for source in ["reddit", "stackoverflow", "github", "news", "patents"]:
                features[f"{source}_problem_count"] = source_counts.get(source, 0)
        else:
            features["matching_problem_count"] = 0
            features["max_problem_similarity"] = 0
            features["avg_problem_similarity"] = 0
            features["high_similarity_count"] = 0
            features["reddit_problem_count"] = 0
            features["stackoverflow_problem_count"] = 0
            features["github_problem_count"] = 0
            features["news_problem_count"] = 0
            features["patents_problem_count"] = 0
            
        return features
    
    def extract_industry_mention_features(self, paper: Dict) -> Dict:
        """
        Extract features related to industry mentions.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of extracted features
        """
        features = {}
        
        if USE_INDUSTRY_MENTIONS:
            # Get industry mentions
            mentions = paper.get("industry_mentions", {})
            
            # Extract counts from different sources
            news_mentions = mentions.get("news", [])
            blog_mentions = mentions.get("blogs", [])
            company_mentions = mentions.get("companies", [])
            
            features["news_mention_count"] = len(news_mentions) if isinstance(news_mentions, list) else 0
            features["blog_mention_count"] = len(blog_mentions) if isinstance(blog_mentions, list) else 0
            features["company_mention_count"] = len(company_mentions) if isinstance(company_mentions, list) else 0
            
            # Total mentions
            features["total_industry_mentions"] = (
                features["news_mention_count"] +
                features["blog_mention_count"] +
                features["company_mention_count"]
            )
            
            # Mention recency
            all_mentions = []
            for source in [news_mentions, blog_mentions, company_mentions]:
                if isinstance(source, list):
                    all_mentions.extend(source)
            
            mention_years = [mention.get("year", 0) for mention in all_mentions if isinstance(mention, dict)]
            if mention_years:
                features["most_recent_mention"] = max(mention_years) if max(mention_years) > 0 else 0
                features["oldest_mention"] = min(mention_years) if min(mention_years) > 0 else 0
                features["mention_years_span"] = features["most_recent_mention"] - features["oldest_mention"]
            else:
                features["most_recent_mention"] = 0
                features["oldest_mention"] = 0
                features["mention_years_span"] = 0
        else:
            features["news_mention_count"] = 0
            features["blog_mention_count"] = 0
            features["company_mention_count"] = 0
            features["total_industry_mentions"] = 0
            features["most_recent_mention"] = 0
            features["oldest_mention"] = 0
            features["mention_years_span"] = 0
            
        return features
    
    def extract_all_features(self, paper: Dict) -> Dict:
        """
        Extract all features for a paper.
        
        Args:
            paper: Paper document from MongoDB
            
        Returns:
            dict: Dictionary of all extracted features
        """
        # Combine all feature types
        features = {}
        features.update(self.extract_basic_features(paper))
        features.update(self.extract_citation_features(paper))
        features.update(self.extract_author_features(paper))
        features.update(self.extract_problem_similarity_features(paper))
        features.update(self.extract_industry_mention_features(paper))
        
        # Add score components from proxy label generation
        score_components = paper.get("score_components", {})
        if score_components:
            features["patent_score"] = score_components.get("patent_citations", 0)
            features["industry_score"] = score_components.get("industry_mentions", 0)
            features["similarity_score"] = score_components.get("problem_similarity", 0)
            features["affiliation_score"] = score_components.get("author_affiliation", 0)
        else:
            features["patent_score"] = 0
            features["industry_score"] = 0
            features["similarity_score"] = 0
            features["affiliation_score"] = 0
        
        # Add commercial score as a feature (can be used for evaluation)
        features["commercial_score"] = paper.get("commercial_score", 0)
        
        # Add LLM scoring if available
        if USE_LLM_SCORING and "llm_score" in paper:
            features["llm_score"] = paper.get("llm_score", 0)
        
        return features
    
    def create_feature_dataset(self, limit: int = 0) -> pd.DataFrame:
        """
        Create feature dataset from all papers with commercial scores.
        
        Args:
            limit: Maximum number of papers to process (0 for all)
            
        Returns:
            pd.DataFrame: DataFrame with features for all papers
        """
        # Get papers with commercial scores
        paper_collection = self.mongo.get_papers_collection()
        query = {"commercial_score": {"$exists": True}}
        
        cursor = paper_collection.find(query)
        if limit > 0:
            cursor = cursor.limit(limit)
            
        papers = list(cursor)
        logger.info(f"Extracting features for {len(papers)} papers")
        
        # Extract features for each paper
        paper_features = []
        
        for paper in tqdm(papers):
            try:
                # Extract features
                features = self.extract_all_features(paper)
                
                # Add paper ID and label
                features["paper_id"] = str(paper["_id"])
                features["commercial_label"] = paper.get("commercial_label", 0)
                
                paper_features.append(features)
                
            except Exception as e:
                logger.error(f"Error extracting features for paper {paper.get('_id')}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(paper_features)
        logger.info(f"Created feature dataset with {len(df)} papers and {len(df.columns)} features")
        
        return df
    
    def save_feature_dataset(self, output_path: str, limit: int = 0) -> str:
        """
        Create and save feature dataset to CSV.
        
        Args:
            output_path: Path to save the CSV file
            limit: Maximum number of papers to process (0 for all)
            
        Returns:
            str: Path to saved CSV file
        """
        # Create dataset
        df = self.create_feature_dataset(limit)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved feature dataset to {output_path}")
        
        return output_path


def main():
    """Main function to extract features and create a dataset."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "commercial_value_features.csv")
    
    extractor = CommercialValueFeatureExtractor()
    extractor.save_feature_dataset(output_path)
    
    
if __name__ == "__main__":
    main()
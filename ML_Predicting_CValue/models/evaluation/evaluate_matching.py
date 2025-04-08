#!/usr/bin/env python3
"""
Evaluation script for the Paper-Problem Matching Model.
Evaluates similarity search quality and match relevance.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from collections import defaultdict

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import SIMILARITY_THRESHOLD, TOP_K_PAPERS, TOP_K_PROBLEMS
from models.matching.similarity_model import SimilarityModel
from models.matching.cross_encoder import CrossEncoder
from data.connectors.mongo_connector import get_connector as get_mongo_connector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatchingEvaluator:
    """
    Evaluator for Paper-Problem Matching system.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.mongo = get_mongo_connector()
        self.mongo.connect()
        
        self.similarity_model = SimilarityModel()
        self.cross_encoder = None  # Lazy load
        
        # Create output directory
        self.output_dir = Path("models/evaluation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_cross_encoder(self):
        """Lazy load the cross-encoder."""
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder()
    
    def evaluate_random_matches(self, n_samples: int = 50, k: int = 10) -> Dict:
        """
        Evaluate matching quality on random papers.
        
        Args:
            n_samples: Number of papers to sample
            k: Number of matches to retrieve per paper
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating matching on {n_samples} random papers")
        
        # Get random papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        cursor = paper_collection.aggregate([
            {"$match": {"scibert_embedding": {"$exists": True}}},
            {"$sample": {"size": n_samples}}
        ])
        
        papers = list(cursor)
        logger.info(f"Sampled {len(papers)} papers")
        
        if not papers:
            logger.error("No papers found with embeddings")
            return {"error": "No papers found"}
        
        # Get matches for each paper
        all_matches = []
        total_matches = 0
        papers_with_matches = 0
        similarity_scores = []
        
        for paper in tqdm(papers, desc="Finding matches"):
            paper_id = paper["_id"]
            
            # Get matches
            matches = self.similarity_model.find_similar_problems(
                paper_id, k=k, threshold=SIMILARITY_THRESHOLD
            )
            
            if matches:
                papers_with_matches += 1
                total_matches += len(matches)
                similarity_scores.extend([m["similarity_score"] for m in matches])
                
                # Add paper info to match
                for match in matches:
                    match["paper_id"] = paper_id
                    match["paper_title"] = paper.get("title", "")
                    match["paper_abstract"] = paper.get("abstract", "")
                    all_matches.append(match)
        
        # Calculate metrics
        avg_matches_per_paper = total_matches / len(papers) if papers else 0
        papers_with_matches_pct = papers_with_matches / len(papers) * 100 if papers else 0
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # Analyze match distribution by source
        source_counts = defaultdict(int)
        for match in all_matches:
            source = match.get("source", "unknown")
            source_counts[source] += 1
        
        # Prepare results
        results = {
            "num_papers": len(papers),
            "papers_with_matches": papers_with_matches,
            "papers_with_matches_pct": papers_with_matches_pct,
            "total_matches": total_matches,
            "avg_matches_per_paper": avg_matches_per_paper,
            "avg_similarity": float(avg_similarity),
            "source_distribution": dict(source_counts)
        }
        
        # Generate visualizations
        self._generate_matching_visualizations(all_matches, results)
        
        # Save matches for manual inspection
        matches_sample = random.sample(all_matches, min(20, len(all_matches)))
        self._save_matches_for_inspection(matches_sample)
        
        return results
    
    def evaluate_cross_encoder_reranking(self, n_samples: int = 20, k: int = 20) -> Dict:
        """
        Evaluate cross-encoder reranking on random papers.
        
        Args:
            n_samples: Number of papers to sample
            k: Number of matches to retrieve per paper
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating cross-encoder reranking on {n_samples} random papers")
        
        # Load cross-encoder
        self._load_cross_encoder()
        
        # Get random papers with embeddings
        paper_collection = self.mongo.get_papers_collection()
        cursor = paper_collection.aggregate([
            {"$match": {"scibert_embedding": {"$exists": True}}},
            {"$sample": {"size": n_samples}}
        ])
        
        papers = list(cursor)
        logger.info(f"Sampled {len(papers)} papers")
        
        if not papers:
            logger.error("No papers found with embeddings")
            return {"error": "No papers found"}
        
        # Evaluate reranking for each paper
        reranking_results = []
        
        for paper in tqdm(papers, desc="Evaluating reranking"):
            paper_id = paper["_id"]
            
            # Get biencoder matches
            bi_matches = self.similarity_model.find_similar_problems(
                paper_id, k=k, threshold=0.5  # Lower threshold to get more matches
            )
            
            if not bi_matches:
                continue
                
            # Rerank with cross-encoder
            reranked_matches = self.cross_encoder.rerank_matches(paper_id, bi_matches.copy())
            
            # Analyze rankings
            bi_order = {m["problem_id"]: i for i, m in enumerate(bi_matches)}
            
            for i, match in enumerate(reranked_matches):
                problem_id = match["problem_id"]
                
                # Calculate rank change
                if problem_id in bi_order:
                    old_rank = bi_order[problem_id]
                    rank_change = old_rank - i
                else:
                    rank_change = None
                
                reranking_results.append({
                    "paper_id": paper_id,
                    "problem_id": problem_id,
                    "bi_encoder_score": match.get("bi_encoder_score", 0),
                    "cross_encoder_score": match.get("cross_encoder_score", 0),
                    "final_score": match.get("similarity_score", 0),
                    "new_rank": i,
                    "old_rank": old_rank if problem_id in bi_order else None,
                    "rank_change": rank_change
                })
        
        # Calculate metrics
        df = pd.DataFrame(reranking_results)
        
        if len(df) == 0:
            logger.warning("No reranking results to analyze")
            return {"error": "No reranking results"}
        
        # Calculate correlation between scores
        bi_cross_corr = df["bi_encoder_score"].corr(df["cross_encoder_score"])
        
        # Calculate average absolute rank change
        avg_abs_rank_change = df["rank_change"].abs().mean()
        
        # Calculate percentage of significant rank changes (more than 3 positions)
        significant_changes = (df["rank_change"].abs() > 3).mean() * 100
        
        # Prepare results
        results = {
            "num_papers": len(papers),
            "num_matches_evaluated": len(df),
            "bi_cross_corr": float(bi_cross_corr),
            "avg_abs_rank_change": float(avg_abs_rank_change),
            "significant_changes_pct": float(significant_changes),
            "avg_bi_encoder_score": float(df["bi_encoder_score"].mean()),
            "avg_cross_encoder_score": float(df["cross_encoder_score"].mean()),
            "avg_final_score": float(df["final_score"].mean())
        }
        
        # Generate visualizations
        self._generate_reranking_visualizations(df, results)
        
        return results
    
    def evaluate_search_quality(self, n_queries: int = 10) -> Dict:
        """
        Evaluate search quality by comparing results for different queries.
        
        Args:
            n_queries: Number of problem queries to test
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating search quality with {n_queries} queries")
        
        # Define test queries (business problems)
        test_queries = [
            "How can we optimize database performance for high-traffic applications?",
            "We need a more efficient algorithm for product recommendations.",
            "Our machine learning model takes too long to train on large datasets.",
            "We need to improve natural language processing for customer support.",
            "Looking for ways to detect fraud in financial transactions.",
            "How to implement secure authentication in a microservices architecture?",
            "We need to optimize our supply chain with predictive analytics.",
            "Need to extract structured data from unstructured documents.",
            "How to implement real-time analytics on streaming data?",
            "Our image recognition system is too slow for real-time applications.",
            "Need to reduce energy consumption in our data centers.",
            "How to implement federated learning for privacy-preserving AI?",
            "Looking for ways to optimize cloud infrastructure costs.",
            "Need better algorithms for anomaly detection in IoT sensor data.",
            "How to implement efficient graph processing for social network analysis?"
        ]
        
        # Randomly select queries if we have more than needed
        if len(test_queries) > n_queries:
            test_queries = random.sample(test_queries, n_queries)
            
        # Evaluate each query
        search_results = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"Query {i+1}/{len(test_queries)}: {query}")
            
            # Find papers for the query
            papers = self.similarity_model.find_papers_for_text(query)
            
            # Analyze results
            result = {
                "query": query,
                "num_results": len(papers),
                "avg_similarity": np.mean([p["similarity_score"] for p in papers]) if papers else 0,
                "avg_commercial_score": np.mean([p.get("commercial_score", 0) for p in papers]) if papers else 0,
                "top_papers": [p["title"] for p in papers[:3]] if papers else []
            }
            
            search_results.append(result)
            
        # Calculate overall metrics
        avg_results_per_query = np.mean([r["num_results"] for r in search_results])
        avg_similarity = np.mean([r["avg_similarity"] for r in search_results])
        avg_commercial_score = np.mean([r["avg_commercial_score"] for r in search_results])
        
        # Prepare results
        results = {
            "num_queries": len(test_queries),
            "avg_results_per_query": float(avg_results_per_query),
            "avg_similarity": float(avg_similarity),
            "avg_commercial_score": float(avg_commercial_score),
            "queries": search_results
        }
        
        # Generate visualizations
        self._generate_search_visualizations(search_results, results)
        
        return results
    
    def _generate_matching_visualizations(self, matches: List[Dict], metrics: Dict) -> None:
        """
        Generate visualizations for random matching evaluation.
        
        Args:
            matches: All matches found
            metrics: Matching metrics
        """
        # Create timestamp for filenames
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create similarity score distribution plot
        similarity_scores = [m["similarity_score"] for m in matches]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(similarity_scores, bins=20, kde=True)
        plt.axvline(x=SIMILARITY_THRESHOLD, color='r', linestyle='--', 
                   label=f'Threshold ({SIMILARITY_THRESHOLD})')
        plt.title("Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        
        scores_path = self.output_dir / f"similarity_scores_{timestamp}.png"
        plt.savefig(scores_path)
        plt.close()
        
        # Create source distribution plot
        source_counts = metrics["source_distribution"]
        
        if source_counts:
            plt.figure(figsize=(12, 8))
            sources = list(source_counts.keys())
            counts = list(source_counts.values())
            
            # Sort by count
            source_data = sorted(zip(sources, counts), key=lambda x: x[1], reverse=True)
            sources, counts = zip(*source_data)
            
            plt.bar(sources, counts)
            plt.title("Matches by Source")
            plt.xlabel("Source")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            source_path = self.output_dir / f"match_sources_{timestamp}.png"
            plt.savefig(source_path)
            plt.close()
        
        # Save metrics to JSON
        metrics_path = self.output_dir / f"matching_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def _generate_reranking_visualizations(self, df: pd.DataFrame, metrics: Dict) -> None:
        """
        Generate visualizations for reranking evaluation.
        
        Args:
            df: DataFrame with reranking results
            metrics: Reranking metrics
        """
        # Create timestamp for filenames
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create score correlation plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df, x="bi_encoder_score", y="cross_encoder_score", alpha=0.6)
        plt.title(f"Bi-Encoder vs Cross-Encoder Scores (r = {metrics['bi_cross_corr']:.3f})")
        plt.xlabel("Bi-Encoder Score")
        plt.ylabel("Cross-Encoder Score")
        plt.grid(alpha=0.3)
        
        corr_path = self.output_dir / f"score_correlation_{timestamp}.png"
        plt.savefig(corr_path)
        plt.close()
        
        # Create rank change distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(df["rank_change"], bins=20, kde=True)
        plt.axvline(x=0, color='r', linestyle='--', label='No Change')
        plt.title("Distribution of Rank Changes")
        plt.xlabel("Rank Change")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        
        rank_path = self.output_dir / f"rank_changes_{timestamp}.png"
        plt.savefig(rank_path)
        plt.close()
        
        # Save metrics to JSON
        metrics_path = self.output_dir / f"reranking_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def _generate_search_visualizations(self, search_results: List[Dict], metrics: Dict) -> None:
        """
        Generate visualizations for search quality evaluation.
        
        Args:
            search_results: Results for each query
            metrics: Search metrics
        """
        # Create timestamp for filenames
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results per query plot
        plt.figure(figsize=(12, 8))
        
        # Truncate query strings for readability
        query_labels = [q["query"][:30] + "..." if len(q["query"]) > 30 else q["query"] 
                       for q in search_results]
        result_counts = [q["num_results"] for q in search_results]
        
        bars = plt.bar(range(len(query_labels)), result_counts)
        plt.xticks(range(len(query_labels)), query_labels, rotation=45, ha='right')
        plt.title("Number of Results per Query")
        plt.xlabel("Query")
        plt.ylabel("Number of Results")
        plt.axhline(y=metrics["avg_results_per_query"], color='r', linestyle='--', 
                   label=f'Average ({metrics["avg_results_per_query"]:.1f})')
        plt.legend()
        plt.tight_layout()
        
        results_path = self.output_dir / f"results_per_query_{timestamp}.png"
        plt.savefig(results_path)
        plt.close()
        
        # Create comparison of similarity and commercial scores
        plt.figure(figsize=(12, 8))
        
        similarity_scores = [q["avg_similarity"] for q in search_results]
        commercial_scores = [q["avg_commercial_score"] for q in search_results]
        
        x = range(len(query_labels))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], similarity_scores, width, label='Similarity Score')
        plt.bar([i + width/2 for i in x], commercial_scores, width, label='Commercial Score')
        
        plt.xticks(x, query_labels, rotation=45, ha='right')
        plt.title("Average Scores per Query")
        plt.xlabel("Query")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        
        scores_path = self.output_dir / f"scores_per_query_{timestamp}.png"
        plt.savefig(scores_path)
        plt.close()
        
        # Save metrics to JSON
        metrics_path = self.output_dir / f"search_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved visualizations to {self.output_dir}")
    
    def _save_matches_for_inspection(self, matches: List[Dict]) -> None:
        """
        Save sample matches for manual inspection.
        
        Args:
            matches: Sample of matches to save
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a detailed report of matches
        report = []
        
        for match in matches:
            report.append({
                "paper_title": match.get("paper_title", ""),
                "paper_abstract": match.get("paper_abstract", "")[:300] + "..." if len(match.get("paper_abstract", "")) > 300 else match.get("paper_abstract", ""),
                "problem_text": match.get("text", ""),
                "problem_source": match.get("source", ""),
                "similarity_score": match.get("similarity_score", 0),
                "paper_id": str(match.get("paper_id", "")),
                "problem_id": str(match.get("problem_id", ""))
            })
        
        # Save the report
        report_path = self.output_dir / f"match_samples_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved {len(report)} matches for inspection to {report_path}")
        
        # Create a more readable HTML version
        html_content = """
        <html>
        <head>
            <title>Match Samples for Inspection</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .match { border: 1px solid #ccc; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
                .score { font-weight: bold; color: #007bff; }
                .paper { background-color: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
                .problem { background-color: #e9ecef; padding: 10px; border-radius: 5px; }
                .source { color: #6c757d; font-style: italic; }
            </style>
        </head>
        <body>
            <h1>Match Samples for Inspection</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="matches">
        """.format(timestamp=timestamp)
        
        for i, match in enumerate(report):
            html_content += """
            <div class="match">
                <h3>Match #{num}</h3>
                <p class="score">Similarity Score: {score:.4f}</p>
                
                <div class="paper">
                    <h4>Paper: {paper_title}</h4>
                    <p>{paper_abstract}</p>
                </div>
                
                <div class="problem">
                    <h4>Problem:</h4>
                    <p>{problem_text}</p>
                    <p class="source">Source: {problem_source}</p>
                </div>
            </div>
            """.format(
                num=i+1,
                score=match["similarity_score"],
                paper_title=match["paper_title"],
                paper_abstract=match["paper_abstract"],
                problem_text=match["problem_text"],
                problem_source=match["problem_source"]
            )
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save the HTML report
        html_path = self.output_dir / f"match_samples_{timestamp}.html"
        with open(html_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Saved HTML report to {html_path}")


def main():
    """Run matching evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Paper-Problem Matching")
    parser.add_argument("--random-matches", action="store_true", help="Evaluate random matches")
    parser.add_argument("--reranking", action="store_true", help="Evaluate cross-encoder reranking")
    parser.add_argument("--search", action="store_true", help="Evaluate search quality")
    parser.add_argument("--samples", type=int, default=50, help="Number of papers to sample")
    parser.add_argument("--queries", type=int, default=10, help="Number of test queries")
    
    args = parser.parse_args()
    
    evaluator = MatchingEvaluator()
    
    if args.random_matches:
        results = evaluator.evaluate_random_matches(n_samples=args.samples)
        print("\nRandom Matching Evaluation:")
        print(f"  Sampled papers: {results['num_papers']}")
        print(f"  Papers with matches: {results['papers_with_matches']} ({results['papers_with_matches_pct']:.1f}%)")
        print(f"  Average matches per paper: {results['avg_matches_per_paper']:.2f}")
        print(f"  Average similarity score: {results['avg_similarity']:.4f}")
        print("  Source distribution:")
        for source, count in sorted(results['source_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {source}: {count}")
    
    if args.reranking:
        results = evaluator.evaluate_cross_encoder_reranking(n_samples=args.samples)
        if "error" not in results:
            print("\nReranking Evaluation:")
            print(f"  Sampled papers: {results['num_papers']}")
            print(f"  Matches evaluated: {results['num_matches_evaluated']}")
            print(f"  Bi-encoder/cross-encoder correlation: {results['bi_cross_corr']:.4f}")
            print(f"  Average absolute rank change: {results['avg_abs_rank_change']:.2f}")
            print(f"  Significant rank changes: {results['significant_changes_pct']:.1f}%")
            print(f"  Average bi-encoder score: {results['avg_bi_encoder_score']:.4f}")
            print(f"  Average cross-encoder score: {results['avg_cross_encoder_score']:.4f}")
        else:
            print(f"\nError in reranking evaluation: {results['error']}")
    
    if args.search:
        results = evaluator.evaluate_search_quality(n_queries=args.queries)
        print("\nSearch Quality Evaluation:")
        print(f"  Test queries: {results['num_queries']}")
        print(f"  Average results per query: {results['avg_results_per_query']:.2f}")
        print(f"  Average similarity score: {results['avg_similarity']:.4f}")
        print(f"  Average commercial score: {results['avg_commercial_score']:.4f}")
        
        for i, query in enumerate(results['queries']):
            print(f"\n  Query {i+1}: {query['query']}")
            print(f"    Results: {query['num_results']}")
            print(f"    Top papers:")
            for j, paper in enumerate(query['top_papers']):
                print(f"      {j+1}. {paper}")
    
    # If no arguments, print help
    if not (args.random_matches or args.reranking or args.search):
        parser.print_help()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Periodic Model Retraining Script for Prospectis ML.
Retrains the commercial value prediction model using updated data and feedback.
"""

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import shutil

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.commercial_value.proxy_label_generator import ProxyLabelGenerator
from models.commercial_value.feature_extraction import CommercialValueFeatureExtractor
from models.commercial_value.cv_model import CommercialValueModel
from models.evaluation.evaluate_cv import CommercialValueEvaluator
from feedback.process_feedback import FeedbackProcessor
from embeddings.index_management import VectorIndexManager
from data.connectors.mongo_connector import get_connector
from utils.logging import setup_logger, log_execution_time
from config.model_config import CV_MODEL_PATH, CV_MODEL_TYPE

# Set up logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"model_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = setup_logger("model_retraining", log_file=log_file, console=True)


class ModelRetrainer:
    """
    Handles model retraining and evaluation.
    """
    
    def __init__(self, model_type: str = CV_MODEL_TYPE, model_path: str = CV_MODEL_PATH):
        """
        Initialize model retrainer.
        
        Args:
            model_type: Type of model to train
            model_path: Path to save model
        """
        self.model_type = model_type
        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up data directory
        self.data_dir = Path(__file__).parent.parent / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create version directory for this training run
        self.version_dir = self.model_dir / f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.version_dir.mkdir(exist_ok=True)
        
        # Set up archive of previous model
        self.archive_dir = self.model_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        # Connect to database
        self.mongo = get_connector()
        self.mongo.connect()
        
        # Initialize components
        self.proxy_generator = ProxyLabelGenerator()
        self.feature_extractor = CommercialValueFeatureExtractor()
        self.feedback_processor = FeedbackProcessor()
        self.index_manager = VectorIndexManager()
        
        # Training config
        self.config = {
            "model_type": model_type,
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
            "version_dir": str(self.version_dir),
            "cross_validation": True,
            "n_folds": 5,
            "evaluate": True,
            "update_vectors": True,
            "process_feedback": True,
            "update_matches": True
        }
    
    def update_proxy_labels(self) -> int:
        """
        Update proxy labels for papers without commercial scores.
        
        Returns:
            int: Number of papers updated
        """
        logger.info("Updating proxy labels for papers without commercial scores")
        start_time = time.time()
        
        results = self.proxy_generator.generate_labels_for_all_papers()
        
        log_execution_time(logger, start_time, "Proxy label generation")
        return len(results)
    
    def extract_features(self, output_path: Path = None) -> str:
        """
        Extract features for training.
        
        Args:
            output_path: Path to save feature dataset
            
        Returns:
            str: Path to feature dataset
        """
        logger.info("Extracting features for model training")
        start_time = time.time()
        
        if output_path is None:
            output_path = self.data_dir / f"commercial_value_features_{datetime.now().strftime('%Y%m%d')}.csv"
        
        output_path = self.feature_extractor.save_feature_dataset(str(output_path))
        
        log_execution_time(logger, start_time, "Feature extraction")
        return output_path
    
    def train_model(self, data_path: str = None, cross_validate: bool = True, 
                  n_folds: int = 5) -> CommercialValueModel:
        """
        Train the commercial value prediction model.
        
        Args:
            data_path: Path to feature dataset
            cross_validate: Whether to perform cross-validation
            n_folds: Number of CV folds
            
        Returns:
            CommercialValueModel: Trained model
        """
        logger.info(f"Training {self.model_type} model")
        start_time = time.time()
        
        # Get or create feature dataset
        if data_path is None or not os.path.exists(data_path):
            data_path = self.extract_features()
        
        # Create versioned model path
        version_model_path = str(self.version_dir / self.model_path.name)
        
        # Create model
        model = CommercialValueModel(model_type=self.model_type, model_path=version_model_path)
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
        
        # Remove non-feature columns
        features_df = df.drop(columns=["paper_id", "commercial_label", "commercial_score"], errors="ignore")
        target = df["commercial_label"]
        
        # Log class distribution
        class_dist = target.value_counts().to_dict()
        logger.info(f"Class distribution: {class_dist}")
        
        # Cross-validation
        if cross_validate:
            logger.info(f"Performing {n_folds}-fold cross-validation")
            cv_results = model.cross_validate(features_df, target, n_folds=n_folds)
            
            logger.info("Cross-validation results:")
            for metric, value in cv_results.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            # Save CV results
            cv_results_path = self.version_dir / "cv_results.json"
            with open(cv_results_path, "w") as f:
                json.dump(cv_results, f, indent=2)
            
            self.config["cv_results"] = cv_results
        
        # Train final model
        logger.info("Training final model on full dataset")
        train_results = model.train(features_df, target)
        
        logger.info("Final model validation metrics:")
        for metric, value in train_results["validation_metrics"].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save training results
        results_path = self.version_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(train_results, f, indent=2)
        
        # Copy dataset to version directory
        dataset_path = self.version_dir / "training_data.csv"
        shutil.copy(data_path, dataset_path)
        
        # Save config
        self.config["training_results"] = train_results
        self.config["dataset_path"] = str(dataset_path)
        self.config["feature_count"] = len(features_df.columns)
        self.config["sample_count"] = len(df)
        self.config["class_distribution"] = {str(k): int(v) for k, v in class_dist.items()}
        
        config_path = self.version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        log_execution_time(logger, start_time, "Model training")
        return model
    
    def evaluate_model(self, model: CommercialValueModel, data_path: str) -> dict:
        """
        Evaluate the trained model.
        
        Args:
            model: Trained model
            data_path: Path to evaluation dataset
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating model")
        start_time = time.time()
        
        # Create evaluator
        evaluator = CommercialValueEvaluator(model_path=model.model_path)
        
        # Evaluate on dataset
        metrics = evaluator.evaluate_dataset(data_path)
        
        # Perform threshold sweep
        sweep_results = evaluator.evaluate_threshold_sweep(data_path)
        
        # Save results to config
        self.config["evaluation"] = {
            "metrics": metrics,
            "threshold_sweep": {
                "optimal_threshold": sweep_results["optimal_threshold"],
                "optimal_metrics": sweep_results["optimal_metrics"]
            }
        }
        
        # Update config file
        config_path = self.version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        log_execution_time(logger, start_time, "Model evaluation")
        return metrics
    
    def archive_current_model(self) -> bool:
        """
        Archive the current model before replacing it.
        
        Returns:
            bool: True if successful
        """
        if not self.model_path.exists():
            logger.warning(f"No existing model found at {self.model_path}, skipping archive")
            return False
        
        logger.info(f"Archiving current model")
        
        try:
            # Create archive filename with timestamp
            archive_name = f"{self.model_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.model_path.suffix}"
            archive_path = self.archive_dir / archive_name
            
            # Copy to archive
            shutil.copy2(self.model_path, archive_path)
            
            # Copy any associated files (feature importances, etc.)
            for related_file in self.model_path.parent.glob(f"{self.model_path.stem}*"):
                if related_file != self.model_path:
                    related_archive = self.archive_dir / f"{related_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{related_file.suffix}"
                    shutil.copy2(related_file, related_archive)
            
            logger.info(f"Archived model to {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive model: {e}")
            return False
    
    def deploy_new_model(self) -> bool:
        """
        Deploy the new model, replacing the current one.
        
        Returns:
            bool: True if successful
        """
        logger.info("Deploying new model")
        
        # Get the new model path from the version directory
        new_model_path = self.version_dir / self.model_path.name
        
        if not new_model_path.exists():
            logger.error(f"New model not found at {new_model_path}")
            return False
        
        try:
            # Copy new model to production path
            shutil.copy2(new_model_path, self.model_path)
            
            # Copy any associated files (feature importances, etc.)
            for related_file in self.version_dir.glob(f"{new_model_path.stem}*"):
                if related_file != new_model_path:
                    related_dest = self.model_path.parent / related_file.name
                    shutil.copy2(related_file, related_dest)
            
            logger.info(f"Deployed new model to {self.model_path}")
            
            # Update model metadata in config
            self.config["deployed"] = True
            self.config["deployment_time"] = datetime.now().isoformat()
            
            config_path = self.version_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    def update_vector_indices(self) -> bool:
        """
        Update vector indices after model retraining.
        
        Returns:
            bool: True if successful
        """
        logger.info("Updating vector indices")
        start_time = time.time()
        
        # Update paper and problem indices
        paper_success = self.index_manager.rebuild_paper_index()
        problem_success = self.index_manager.rebuild_problem_index()
        
        # Optimize indices
        if paper_success and problem_success:
            self.index_manager.optimize_indices()
        
        log_execution_time(logger, start_time, "Vector index update")
        return paper_success and problem_success
    
    def process_feedback(self) -> dict:
        """
        Process user feedback to improve model predictions.
        
        Returns:
            dict: Processing results
        """
        logger.info("Processing user feedback")
        start_time = time.time()
        
        results = self.feedback_processor.process_all_feedback()
        
        log_execution_time(logger, start_time, "Feedback processing")
        return results
    
    def update_matches(self) -> Tuple[int, int]:
        """
        Update paper-problem matches.
        
        Returns:
            tuple: (total_papers, total_matches)
        """
        logger.info("Updating paper-problem matches")
        start_time = time.time()
        
        from models.matching.similarity_model import SimilarityModel
        similarity_model = SimilarityModel()
        
        papers, matches = similarity_model.update_all_matches()
        
        log_execution_time(logger, start_time, "Match update")
        return papers, matches
    
    def retrain(self, cross_validate: bool = True, n_folds: int = 5, 
              evaluate: bool = True, update_vectors: bool = True,
              process_feedback: bool = True, update_matches: bool = True) -> dict:
        """
        Run the full model retraining pipeline.
        
        Args:
            cross_validate: Whether to perform cross-validation
            n_folds: Number of CV folds
            evaluate: Whether to evaluate the model
            update_vectors: Whether to update vector indices
            process_feedback: Whether to process user feedback
            update_matches: Whether to update paper-problem matches
            
        Returns:
            dict: Retraining results
        """
        logger.info(f"Starting model retraining: {self.model_type}")
        overall_start_time = time.time()
        
        # Update config
        self.config.update({
            "cross_validation": cross_validate,
            "n_folds": n_folds,
            "evaluate": evaluate,
            "update_vectors": update_vectors,
            "process_feedback": process_feedback,
            "update_matches": update_matches
        })
        
        # Process user feedback if requested
        if process_feedback:
            feedback_results = self.process_feedback()
            self.config["feedback_results"] = feedback_results
        
        # Update proxy labels
        updated_count = self.update_proxy_labels()
        self.config["papers_with_proxy_labels"] = updated_count
        
        # Extract features
        data_path = self.extract_features()
        
        # Archive current model
        self.archive_current_model()
        
        # Train model
        model = self.train_model(
            data_path=data_path,
            cross_validate=cross_validate,
            n_folds=n_folds
        )
        
        # Evaluate if requested
        if evaluate:
            metrics = self.evaluate_model(model, data_path)
            self.config["final_metrics"] = metrics
        
        # Deploy new model
        self.deploy_new_model()
        
        # Update vector indices if requested
        if update_vectors:
            vector_success = self.update_vector_indices()
            self.config["vector_indices_updated"] = vector_success
        
        # Update paper-problem matches if requested
        if update_matches:
            papers, matches = self.update_matches()
            self.config["matches_updated"] = {
                "papers": papers,
                "matches": matches
            }
        
        # Create summary report
        summary = {
            "model_type": self.model_type,
            "timestamp": self.config["timestamp"],
            "version_dir": str(self.version_dir),
            "dataset_size": self.config.get("sample_count"),
            "class_distribution": self.config.get("class_distribution"),
            "papers_with_proxy_labels": updated_count,
            "deployed": self.config.get("deployed", False),
            "execution_time": time.time() - overall_start_time
        }
        
        # Add metrics if available
        if "final_metrics" in self.config:
            summary["metrics"] = self.config["final_metrics"]
        
        # Save final summary
        summary_path = self.version_dir / "retraining_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        log_execution_time(logger, overall_start_time, "Full model retraining")
        return summary


def main():
    """Run model retraining from command line."""
    parser = argparse.ArgumentParser(description="Retrain Commercial Value Prediction Model")
    parser.add_argument("--model-type", type=str, default=CV_MODEL_TYPE,
                       choices=["xgboost", "lightgbm", "neural", "ensemble"],
                       help="Type of model to train")
    parser.add_argument("--model-path", type=str, default=CV_MODEL_PATH,
                      help="Path to save/replace model")
    parser.add_argument("--no-cv", action="store_true",
                      help="Skip cross-validation")
    parser.add_argument("--no-eval", action="store_true",
                      help="Skip evaluation")
    parser.add_argument("--no-vectors", action="store_true",
                      help="Skip vector index updates")
    parser.add_argument("--no-feedback", action="store_true",
                      help="Skip feedback processing")
    parser.add_argument("--no-matches", action="store_true",
                      help="Skip match updates")
    parser.add_argument("--folds", type=int, default=5,
                      help="Number of CV folds")
    
    args = parser.parse_args()
    
    # Create retrainer
    retrainer = ModelRetrainer(
        model_type=args.model_type,
        model_path=args.model_path
    )
    
    # Run retraining
    summary = retrainer.retrain(
        cross_validate=not args.no_cv,
        n_folds=args.folds,
        evaluate=not args.no_eval,
        update_vectors=not args.no_vectors,
        process_feedback=not args.no_feedback,
        update_matches=not args.no_matches
    )
    
    # Print summary
    print("\nRetraining Summary:")
    print(f"  Model type: {summary['model_type']}")
    print(f"  Version directory: {summary['version_dir']}")
    print(f"  Dataset size: {summary['dataset_size']} samples")
    
    if "class_distribution" in summary:
        print("  Class distribution:")
        for label, count in summary["class_distribution"].items():
            print(f"    {label}: {count}")
    
    if "metrics" in summary:
        print("  Model metrics:")
        for metric, value in summary["metrics"].items():
            print(f"    {metric}: {value:.4f}")
    
    print(f"  Successfully deployed: {summary.get('deployed', False)}")
    print(f"  Total execution time: {summary['execution_time']:.2f} seconds")


if __name__ == "__main__":
    main()
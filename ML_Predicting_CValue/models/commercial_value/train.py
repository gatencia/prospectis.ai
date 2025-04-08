#!/usr/bin/env python3
"""
Training script for Commercial Value Prediction Model.
Trains and evaluates models for predicting commercial value of research papers.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import CV_MODEL_TYPE, CV_MODEL_PATH, CV_FOLDS
from models.commercial_value.cv_model import CommercialValueModel
from models.commercial_value.feature_extraction import CommercialValueFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(data_path=None, model_type=CV_MODEL_TYPE, model_path=CV_MODEL_PATH,
               cross_validate=True, n_folds=CV_FOLDS, limit=0):
    """
    Train and evaluate the commercial value prediction model.
    
    Args:
        data_path: Path to feature dataset CSV (or None to extract features)
        model_type: Type of model to train
        model_path: Path to save model
        cross_validate: Whether to perform cross-validation
        n_folds: Number of CV folds
        limit: Maximum number of papers to process (0 for all)
    """
    # Create output directory
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create feature dataset if not provided
    if data_path is None or not os.path.exists(data_path):
        logger.info("Extracting features from papers")
        extractor = CommercialValueFeatureExtractor()
        data_path = os.path.join(model_dir, "commercial_value_features.csv")
        extractor.save_feature_dataset(data_path, limit=limit)
    
    # Load feature dataset
    logger.info(f"Loading feature dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Drop non-feature columns
    feature_df = df.drop(columns=["paper_id", "commercial_label", "commercial_score"], errors="ignore")
    target = df["commercial_label"]
    
    # Log dataset info
    logger.info(f"Dataset shape: {feature_df.shape}")
    logger.info(f"Class distribution: {target.value_counts().to_dict()}")
    
    # Create model
    logger.info(f"Creating {model_type} model")
    model = CommercialValueModel(model_type=model_type, model_path=model_path)
    
    # Cross-validation
    if cross_validate:
        logger.info(f"Performing {n_folds}-fold cross-validation")
        cv_results = model.cross_validate(feature_df, target, n_folds=n_folds)
        
        logger.info("Cross-validation results:")
        for metric, value in cv_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save CV results
        cv_results_path = os.path.join(model_dir, "cv_results.json")
        with open(cv_results_path, "w") as f:
            json.dump(cv_results, f, indent=2)
    
    # Train final model
    logger.info("Training final model on full dataset")
    train_results = model.train(feature_df, target)
    
    logger.info("Final model validation metrics:")
    for metric, value in train_results["validation_metrics"].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save model path to log
    logger.info(f"Model saved to {model_path}")
    
    # Create a simple report
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "model_path": model_path,
        "dataset": {
            "path": data_path,
            "n_samples": len(df),
            "n_features": len(feature_df.columns),
            "class_distribution": target.value_counts().to_dict()
        },
        "validation_metrics": train_results["validation_metrics"]
    }
    
    if cross_validate:
        report["cross_validation"] = cv_results
    
    # Save report
    report_path = os.path.join(model_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {report_path}")
    
    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Commercial Value Prediction Model")
    parser.add_argument("--data", type=str, help="Path to feature dataset CSV")
    parser.add_argument("--model-type", type=str, default=CV_MODEL_TYPE, 
                       choices=["xgboost", "lightgbm", "neural", "ensemble"],
                       help="Type of model to train")
    parser.add_argument("--model-path", type=str, default=CV_MODEL_PATH,
                      help="Path to save model")
    parser.add_argument("--cross-validate", action="store_true",
                      help="Perform cross-validation")
    parser.add_argument("--folds", type=int, default=CV_FOLDS,
                      help="Number of CV folds")
    parser.add_argument("--limit", type=int, default=0,
                      help="Maximum number of papers to process (0 for all)")
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_path=args.data,
        model_type=args.model_type,
        model_path=args.model_path,
        cross_validate=args.cross_validate,
        n_folds=args.folds,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
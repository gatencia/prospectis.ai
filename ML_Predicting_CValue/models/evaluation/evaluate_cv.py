#!/usr/bin/env python3
"""
Evaluation script for the Commercial Value prediction model.
Evaluates model performance and generates evaluation reports.
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
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score, accuracy_score
)

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import CV_MODEL_PATH
from models.commercial_value.cv_model import CommercialValueModel
from models.commercial_value.feature_extraction import CommercialValueFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommercialValueEvaluator:
    """
    Evaluator for Commercial Value prediction model.
    """
    
    def __init__(self, model_path: str = CV_MODEL_PATH):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.model = CommercialValueModel(model_path=model_path)
        self.model.load()
        
        self.feature_extractor = CommercialValueFeatureExtractor()
        
        # Create output directory
        self.output_dir = Path(model_path).parent / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_dataset(self, data_path: str) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_path: Path to dataset CSV
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating model on dataset: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract features and labels
        if "paper_id" in df.columns:
            X = df.drop(columns=["paper_id", "commercial_label", "commercial_score"], errors="ignore")
        else:
            X = df.drop(columns=["commercial_label", "commercial_score"], errors="ignore")
            
        y_true = df["commercial_label"].values
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        # Generate evaluation report
        self._generate_report(df, y_true, y_pred, y_prob, metrics)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            dict: Metrics
        """
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1"] = f1_score(y_true, y_pred)
        
        # ROC and PR metrics
        if len(np.unique(y_true)) > 1:  # Only if we have both classes
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Add precision and recall for the positive class
        if "1" in report:
            metrics["precision"] = report["1"]["precision"]
            metrics["recall"] = report["1"]["recall"]
        else:
            metrics["precision"] = report["1.0"]["precision"]
            metrics["recall"] = report["1.0"]["recall"]
        
        return metrics
    
    def _generate_report(self, df: pd.DataFrame, y_true: np.ndarray, 
                        y_pred: np.ndarray, y_prob: np.ndarray, 
                        metrics: Dict) -> None:
        """
        Generate evaluation report and visualizations.
        
        Args:
            df: Dataset
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            metrics: Evaluation metrics
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics_path = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to: {metrics_path}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        cm_path = self.output_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(cm_path)
        plt.close()
        
        logger.info(f"Saved confusion matrix to: {cm_path}")
        
        # Generate ROC curve
        if len(np.unique(y_true)) > 1:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            roc_path = self.output_dir / f"roc_curve_{timestamp}.png"
            plt.savefig(roc_path)
            plt.close()
            
            logger.info(f"Saved ROC curve to: {roc_path}")
            
            # Generate Precision-Recall curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall, precision, label=f"PR Curve (AP = {metrics['pr_auc']:.3f})")
            plt.axhline(y=sum(y_true) / len(y_true), color="r", linestyle="--", 
                       label=f"Baseline (Prevalence = {sum(y_true) / len(y_true):.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            pr_path = self.output_dir / f"pr_curve_{timestamp}.png"
            plt.savefig(pr_path)
            plt.close()
            
            logger.info(f"Saved Precision-Recall curve to: {pr_path}")
        
        # Generate score distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=pd.DataFrame({
                "Score": y_prob,
                "Label": y_true.astype(str)
            }),
            x="Score",
            hue="Label",
            bins=30,
            alpha=0.6
        )
        plt.title("Distribution of Prediction Scores by Label")
        plt.xlabel("Commercial Value Score")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        dist_path = self.output_dir / f"score_distribution_{timestamp}.png"
        plt.savefig(dist_path)
        plt.close()
        
        logger.info(f"Saved score distribution to: {dist_path}")
        
        # Create feature importance plot if available
        feature_imp_path = Path(self.model_path).with_name("feature_importances.json")
        if feature_imp_path.exists():
            with open(feature_imp_path, "r") as f:
                feature_importances = json.load(f)
            
            # Get top 20 features
            top_features = dict(sorted(feature_importances.items(), 
                                       key=lambda x: x[1], reverse=True)[:20])
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(list(top_features.keys()), list(top_features.values()))
            plt.xlabel("Importance")
            plt.title("Top 20 Feature Importances")
            plt.gca().invert_yaxis()  # Display highest importance at the top
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            imp_path = self.output_dir / f"feature_importance_{timestamp}.png"
            plt.savefig(imp_path)
            plt.close()
            
            logger.info(f"Saved feature importance plot to: {imp_path}")
        
        # Generate detailed report
        report = {
            "timestamp": timestamp,
            "model_path": str(self.model_path),
            "metrics": metrics,
            "dataset_info": {
                "num_samples": len(df),
                "class_distribution": {
                    "0": int((y_true == 0).sum()),
                    "1": int((y_true == 1).sum())
                },
                "class_balance": f"1: {(y_true == 1).sum() / len(y_true):.2%}"
            },
            "feature_info": {
                "num_features": len(df.columns) - 3 if "paper_id" in df.columns else len(df.columns) - 2,
                "feature_columns": list(df.columns)
            },
            "evaluation_artifacts": {
                "metrics_json": str(metrics_path),
                "confusion_matrix": str(cm_path),
                "roc_curve": str(roc_path) if len(np.unique(y_true)) > 1 else None,
                "pr_curve": str(pr_path) if len(np.unique(y_true)) > 1 else None,
                "score_distribution": str(dist_path),
                "feature_importance": str(imp_path) if feature_imp_path.exists() else None
            }
        }
        
        # Save detailed report
        report_path = self.output_dir / f"evaluation_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved evaluation report to: {report_path}")
    
    def evaluate_threshold_sweep(self, data_path: str) -> Dict:
        """
        Evaluate model with different decision thresholds.
        
        Args:
            data_path: Path to dataset CSV
            
        Returns:
            dict: Metrics for different thresholds
        """
        logger.info(f"Evaluating threshold sweep on dataset: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Extract features and labels
        if "paper_id" in df.columns:
            X = df.drop(columns=["paper_id", "commercial_label", "commercial_score"], errors="ignore")
        else:
            X = df.drop(columns=["commercial_label", "commercial_score"], errors="ignore")
            
        y_true = df["commercial_label"].values
        
        # Get prediction probabilities
        y_prob = self.model.predict_proba(X)
        
        # Test different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            if sum(y_pred) > 0 and sum(y_pred) < len(y_pred):  # Ensure we have both classes
                precision = sum((y_pred == 1) & (y_true == 1)) / sum(y_pred)
                recall = sum((y_pred == 1) & (y_true == 1)) / sum(y_true)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = 0 if sum(y_pred) == 0 else 1
                recall = 0 if sum(y_pred) == 0 else sum(y_true) / len(y_true)
                f1 = 0
                
            accuracy = sum(y_pred == y_true) / len(y_true)
            
            results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "positive_rate": sum(y_pred) / len(y_pred)
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot results
        plt.figure(figsize=(12, 8))
        plt.plot(results_df["threshold"], results_df["precision"], "b-", label="Precision")
        plt.plot(results_df["threshold"], results_df["recall"], "r-", label="Recall")
        plt.plot(results_df["threshold"], results_df["f1"], "g-", label="F1 Score")
        plt.plot(results_df["threshold"], results_df["accuracy"], "k-", label="Accuracy")
        plt.plot(results_df["threshold"], results_df["positive_rate"], "m--", label="Positive Rate")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metrics vs. Threshold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        threshold_path = self.output_dir / f"threshold_sweep_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(threshold_path)
        plt.close()
        
        logger.info(f"Saved threshold sweep plot to: {threshold_path}")
        
        # Find optimal threshold for F1
        optimal_idx = results_df["f1"].idxmax()
        optimal_threshold = results_df.loc[optimal_idx, "threshold"]
        optimal_metrics = results_df.loc[optimal_idx].to_dict()
        
        logger.info(f"Optimal threshold for F1: {optimal_threshold:.2f} (F1 = {optimal_metrics['f1']:.4f})")
        
        # Save results
        results_path = self.output_dir / f"threshold_sweep_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Saved threshold sweep results to: {results_path}")
        
        return {
            "threshold_sweep": results_df.to_dict("records"),
            "optimal_threshold": optimal_threshold,
            "optimal_metrics": optimal_metrics
        }


def main():
    """Run model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Commercial Value prediction model")
    parser.add_argument("--model", type=str, default=CV_MODEL_PATH, help="Path to model file")
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation dataset CSV")
    parser.add_argument("--threshold-sweep", action="store_true", help="Perform threshold sweep analysis")
    
    args = parser.parse_args()
    
    evaluator = CommercialValueEvaluator(model_path=args.model)
    
    # Evaluate on dataset
    metrics = evaluator.evaluate_dataset(args.data)
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Perform threshold sweep if requested
    if args.threshold_sweep:
        sweep_results = evaluator.evaluate_threshold_sweep(args.data)
        
        print("\nOptimal Threshold Analysis:")
        print(f"  Optimal threshold: {sweep_results['optimal_threshold']:.2f}")
        print(f"  F1 score: {sweep_results['optimal_metrics']['f1']:.4f}")
        print(f"  Precision: {sweep_results['optimal_metrics']['precision']:.4f}")
        print(f"  Recall: {sweep_results['optimal_metrics']['recall']:.4f}")


if __name__ == "__main__":
    main()
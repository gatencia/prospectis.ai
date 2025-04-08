"""
Metrics utilities for Prospectis ML system.
Provides functions for evaluating ML models and measuring system performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    ndcg_score, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        dict: Metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    
    # Add probability-based metrics if available
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Metrics
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mae": np.mean(np.abs(y_true - y_pred))
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> Dict[str, float]:
    """
    Calculate metrics for ranking problems.
    
    Args:
        y_true: True relevance scores
        y_score: Predicted relevance scores
        k: Cutoff for nDCG
        
    Returns:
        dict: Metrics
    """
    # Reshape for sklearn metrics if needed
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
        y_score = y_score.reshape(1, -1)
    
    metrics = {}
    
    # Calculate nDCG at k
    if y_true.shape[1] >= k:
        metrics[f"ndcg@{k}"] = ndcg_score(y_true, y_score, k=k)
    
    # Calculate nDCG at different cutoffs
    for cutoff in [5, 10, 20]:
        if y_true.shape[1] >= cutoff:
            metrics[f"ndcg@{cutoff}"] = ndcg_score(y_true, y_score, k=cutoff)
    
    # Calculate mean average precision
    binary_relevance = (y_true > 0).astype(int)
    
    # Make sure there's at least one relevant item
    if np.sum(binary_relevance) > 0:
        # Calculate average precision for each query
        avg_precisions = []
        for i in range(y_true.shape[0]):
            if np.sum(binary_relevance[i]) > 0:  # Skip queries with no relevant items
                # Get sorted indices
                sorted_indices = np.argsort(y_score[i])[::-1]
                
                # Calculate precision at each position
                precisions = []
                num_relevant = 0
                
                for j, idx in enumerate(sorted_indices):
                    if binary_relevance[i, idx] == 1:
                        num_relevant += 1
                        precisions.append(num_relevant / (j + 1))
                
                # Average precision for this query
                avg_precision = np.mean(precisions) if precisions else 0
                avg_precisions.append(avg_precision)
        
        # Mean average precision across all queries
        if avg_precisions:
            metrics["map"] = np.mean(avg_precisions)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: Optional[List[str]] = None,
                         output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (optional)
        output_path: Path to save plot (optional)
        
    Returns:
        np.ndarray: Confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    if labels:
        plt.xticks(np.arange(len(labels)) + 0.5, labels)
        plt.yticks(np.arange(len(labels)) + 0.5, labels)
    
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return cm


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                 output_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        output_path: Path to save plot (optional)
        
    Returns:
        tuple: (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return fpr, tpr, thresholds


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                              output_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        output_path: Path to save plot (optional)
        
    Returns:
        tuple: (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.3f})")
    plt.axhline(y=np.mean(y_true), color="r", linestyle="--", 
               label=f"Baseline (Prevalence = {np.mean(y_true):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return precision, recall, thresholds


def plot_score_distribution(y_true: np.ndarray, y_prob: np.ndarray,
                          output_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot distribution of prediction scores by class.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        output_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    data = pd.DataFrame({
        "Score": y_prob,
        "Class": y_true.astype(str)
    })
    
    sns.histplot(data=data, x="Score", hue="Class", bins=30, alpha=0.6)
    plt.title("Distribution of Prediction Scores by Class")
    plt.xlabel("Predicted Score")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                          top_n: int = 20, sort: bool = True,
                          output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Plot feature importances.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_n: Number of top features to show
        sort: Whether to sort features by importance
        output_path: Path to save plot (optional)
        
    Returns:
        pd.DataFrame: DataFrame with feature importances
    """
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    
    # Sort if requested
    if sort:
        df = df.sort_values("Importance", ascending=False)
    
    # Take top N features
    if top_n > 0:
        df = df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, top_n * 0.4))
    bars = plt.barh(df["Feature"], df["Importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return df


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                         metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        tuple: (optimal_threshold, metrics_at_optimal)
    """
    # Try different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    best_metric = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle edge cases (all predictions same class)
        if len(np.unique(y_pred)) == 1:
            if np.unique(y_pred)[0] == 1:
                # All predicted positive
                precision = np.mean(y_true)
                recall = 1.0
            else:
                # All predicted negative
                precision = 0.0
                recall = 0.0
        else:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Check if this threshold is better
        current_metric = {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }[metric]
        
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
    
    # Get metrics at optimal threshold
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    metrics = binary_classification_metrics(y_true, y_pred_optimal)
    
    return best_threshold, metrics


def calculate_similarity_metrics(
    query_embeddings: np.ndarray, 
    doc_embeddings: np.ndarray,
    relevance: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for embedding-based similarity search.
    
    Args:
        query_embeddings: Query embeddings (n_queries, dim)
        doc_embeddings: Document embeddings (n_docs, dim)
        relevance: Relevance matrix (n_queries, n_docs)
        
    Returns:
        dict: Metrics
    """
    n_queries, query_dim = query_embeddings.shape
    n_docs, doc_dim = doc_embeddings.shape
    
    if query_dim != doc_dim:
        raise ValueError(f"Embedding dimensions don't match: {query_dim} vs {doc_dim}")
    
    if relevance.shape != (n_queries, n_docs):
        raise ValueError(f"Relevance matrix shape {relevance.shape} doesn't match embeddings")
    
    # Calculate cosine similarity for each query-document pair
    # Normalize embeddings
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    query_normalized = query_embeddings / query_norms
    doc_normalized = doc_embeddings / doc_norms
    
    # Calculate similarities
    similarity_matrix = np.dot(query_normalized, doc_normalized.T)
    
    # Calculate ranking metrics
    ndcg_values = []
    precision_at_k_values = []
    recall_at_k_values = []
    
    for i in range(n_queries):
        # Sort documents by similarity
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        
        # Get relevance scores in sorted order
        sorted_relevance = relevance[i, sorted_indices]
        
        # Calculate nDCG@10
        if len(sorted_relevance) >= 10:
            ndcg = ndcg_score(relevance[i:i+1], similarity_matrix[i:i+1], k=10)
            ndcg_values.append(ndcg)
        
        # Calculate precision and recall at k=10
        if np.sum(relevance[i]) > 0:  # Only if there are relevant documents
            # Binary relevance
            binary_relevance = (relevance[i] > 0).astype(int)
            binary_sorted = binary_relevance[sorted_indices]
            
            # Precision@10
            if len(binary_sorted) >= 10:
                precision_at_10 = np.mean(binary_sorted[:10])
                precision_at_k_values.append(precision_at_10)
            
            # Recall@10
            if len(binary_sorted) >= 10:
                recall_at_10 = np.sum(binary_sorted[:10]) / np.sum(binary_relevance)
                recall_at_k_values.append(recall_at_10)
    
    # Aggregate metrics
    metrics = {}
    
    if ndcg_values:
        metrics["ndcg@10"] = np.mean(ndcg_values)
    
    if precision_at_k_values:
        metrics["precision@10"] = np.mean(precision_at_k_values)
    
    if recall_at_k_values:
        metrics["recall@10"] = np.mean(recall_at_k_values)
    
    return metrics
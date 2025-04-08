#!/usr/bin/env python3
"""
Commercial Value Prediction Model.
Implements models for predicting the commercial value of research papers.
Supports different model types (XGBoost, LightGBM, neural, ensemble).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
from pathlib import Path
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    average_precision_score, roc_auc_score, ndcg_score
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root to path to allow imports from any directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.model_config import (
    CV_MODEL_TYPE, CV_MODEL_PATH, TRAIN_TEST_SPLIT, 
    CV_FOLDS, MAX_ITERATIONS, LEARNING_RATE, EARLY_STOPPING_ROUNDS,
    METRICS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleNeuralNet(nn.Module):
    """Simple neural network for commercial value prediction."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class PaperDataset(Dataset):
    """Dataset for paper features."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix
            labels: Target labels
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CommercialValueModel:
    """Model for predicting commercial value of research papers."""
    
    def __init__(self, model_type: str = CV_MODEL_TYPE, model_path: str = CV_MODEL_PATH):
        """
        Initialize the model.
        
        Args:
            model_type: Type of model ("xgboost", "lightgbm", "neural", "ensemble")
            model_path: Path to save/load model
        """
        self.model_type = model_type.lower()
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set device for neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """
        Get feature names from data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            list: List of feature names
        """
        if isinstance(X, pd.DataFrame):
            return list(X.columns)
        else:
            return [f"feature_{i}" for i in range(X.shape[1])]
    
    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Prepare data for model input.
        
        Args:
            X: Feature data
            
        Returns:
            np.ndarray: Prepared feature matrix
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = self._get_feature_names(X)
            X = X.to_numpy()
        
        return X
    
    def _create_model(self, input_dim: int) -> Any:
        """
        Create a new model instance.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Model instance
        """
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=LEARNING_RATE,
                max_depth=5,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                scale_pos_weight=1,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=LEARNING_RATE,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=42
            )
        elif self.model_type == "neural":
            return SimpleNeuralNet(input_dim).to(self.device)
        elif self.model_type == "ensemble":
            return {
                "xgboost": xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=LEARNING_RATE,
                    max_depth=5,
                    objective="binary:logistic",
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss"
                ),
                "random_forest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                             X_val: np.ndarray, y_val: np.ndarray) -> SimpleNeuralNet:
        """
        Train neural network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            SimpleNeuralNet: Trained model
        """
        # Create model
        model = SimpleNeuralNet(X_train.shape[1]).to(self.device)
        
        # Create datasets
        train_dataset = PaperDataset(X_train, y_train)
        val_dataset = PaperDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Define optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = EARLY_STOPPING_ROUNDS
        patience_counter = 0
        
        for epoch in range(MAX_ITERATIONS):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    y_pred = model(batch_X)
                    loss = criterion(y_pred, batch_y)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{MAX_ITERATIONS}, "
                            f"Train Loss: {train_loss:.4f}, "
                            f"Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_path.with_suffix(".pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(self.model_path.with_suffix(".pt")))
        return model
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], 
             eval_metric: str = "auc") -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature data
            y: Target labels
            eval_metric: Evaluation metric for early stopping
            
        Returns:
            dict: Training results
        """
        # Prepare data
        X = self._prepare_data(X)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create model
        if self.model is None:
            self.model = self._create_model(X_train_scaled.shape[1])
        
        # Train model
        if self.model_type == "xgboost":
            # Create evaluation set
            eval_set = [(X_val_scaled, y_val)]
            
            # Train model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=True
            )
            
            # Save feature importances
            feature_importances = self.model.feature_importances_
            
        elif self.model_type == "lightgbm":
            # Create evaluation set
            eval_set = [(X_val_scaled, y_val)]
            
            # Train model
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                eval_metric=eval_metric,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=10
            )
            
            # Save feature importances
            feature_importances = self.model.feature_importances_
            
        elif self.model_type == "neural":
            # Train neural network
            self.model = self._train_neural_network(
                X_train_scaled, y_train, X_val_scaled, y_val
            )
            
            # Neural networks don't have feature importances
            feature_importances = None
            
        elif self.model_type == "ensemble":
            # Train each model in the ensemble
            for name, model in self.model.items():
                logger.info(f"Training {name} model")
                
                if name == "xgboost":
                    eval_set = [(X_val_scaled, y_val)]
                    model.fit(
                        X_train_scaled, y_train,
                        eval_set=eval_set,
                        eval_metric=eval_metric,
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        verbose=False
                    )
                else:
                    model.fit(X_train_scaled, y_train)
            
            # Use XGBoost feature importances
            feature_importances = self.model["xgboost"].feature_importances_
        
        # Save model
        self.save()
        
        # Evaluate on validation set
        metrics = self.evaluate(X_val_scaled, y_val)
        
        # Save feature importances
        if feature_importances is not None and self.feature_names is not None:
            importance_dict = {
                name: float(imp) 
                for name, imp in zip(self.feature_names, feature_importances)
            }
            
            # Sort by importance
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Save to file
            with open(self.model_path.with_name("feature_importances.json"), "w") as f:
                json.dump(importance_dict, f, indent=2)
        
        return {
            "validation_metrics": metrics,
            "feature_importances": importance_dict if feature_importances is not None else None
        }
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature data
            
        Returns:
            np.ndarray: Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        X = self._prepare_data(X)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == "neural":
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy()
            
            # Convert to binary
            return (y_pred > 0.5).astype(int).flatten()
        elif self.model_type == "ensemble":
            # Get predictions from each model
            predictions = []
            
            for name, model in self.model.items():
                if name == "xgboost":
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_scaled)
                
                predictions.append(pred)
            
            # Average predictions (simple ensemble)
            ensemble_pred = np.mean(predictions, axis=0)
            return (ensemble_pred > 0.5).astype(int)
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Feature data
            
        Returns:
            np.ndarray: Probability predictions [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        X = self._prepare_data(X)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == "neural":
            # Convert to PyTorch tensor
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_tensor).cpu().numpy()
            
            return y_pred.flatten()
        elif self.model_type == "ensemble":
            # Get predictions from each model
            predictions = []
            
            for name, model in self.model.items():
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_scaled)[:, 1]
                else:
                    pred = model.predict(X_scaled)
                
                predictions.append(pred)
            
            # Average predictions (simple ensemble)
            return np.mean(predictions, axis=0)
        else:
            return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X: Feature data
            y: True labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = np.mean(y_pred == y)
        metrics["precision"] = precision_score(y, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y, y_pred)
        metrics["f1"] = f1_score(y, y_pred)
        
        # Ranking metrics
        if len(np.unique(y)) > 1:  # Only if we have both classes
            metrics["auc"] = roc_auc_score(y, y_proba)
            metrics["average_precision"] = average_precision_score(y, y_proba)
            
            # NDCG (treat predictions as ranking scores)
            y_true_reshaped = y.reshape(1, -1)
            y_pred_reshaped = y_proba.reshape(1, -1)
            metrics["ndcg"] = ndcg_score(y_true_reshaped, y_pred_reshaped)
        
        return metrics
    
    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Union[pd.Series, np.ndarray], n_folds: int = CV_FOLDS) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature data
            y: True labels
            n_folds: Number of folds
            
        Returns:
            dict: Cross-validation results
        """
        # Prepare data
        X = self._prepare_data(X)
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation
        cv_results = {}
        
        if self.model_type == "neural":
            # Neural network cross-validation requires custom implementation
            # This is a simplified approach
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                logger.info(f"Training fold {fold+1}/{n_folds}")
                
                # Split data
                X_train_fold = X_scaled[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_scaled[val_idx]
                y_val_fold = y[val_idx]
                
                # Train model for this fold
                model_fold = self._train_neural_network(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )
                
                # Evaluate
                model_fold.eval()
                X_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    y_pred = model_fold(X_tensor).cpu().numpy().flatten()
                    y_pred_binary = (y_pred > 0.5).astype(int)
                
                # Calculate metrics
                metrics = {}
                metrics["accuracy"] = np.mean(y_pred_binary == y_val_fold)
                metrics["precision"] = precision_score(y_val_fold, y_pred_binary, zero_division=0)
                metrics["recall"] = recall_score(y_val_fold, y_pred_binary)
                metrics["f1"] = f1_score(y_val_fold, y_pred_binary)
                
                if len(np.unique(y_val_fold)) > 1:
                    metrics["auc"] = roc_auc_score(y_val_fold, y_pred)
                    metrics["average_precision"] = average_precision_score(y_val_fold, y_pred)
                
                fold_metrics.append(metrics)
            
            # Average results across folds
            cv_results = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in fold_metrics[0].keys()
            }
            
            # Add std for each metric
            for metric in fold_metrics[0].keys():
                cv_results[f"{metric}_std"] = np.std([fold[metric] for fold in fold_metrics])
                
        elif self.model_type == "ensemble":
            # Cross-validate each model separately
            ensemble_results = {}
            
            for name, model_class in [
                ("xgboost", xgb.XGBClassifier),
                ("random_forest", RandomForestClassifier)
            ]:
                logger.info(f"Cross-validating {name} model")
                
                # Create a fresh model for CV
                if name == "xgboost":
                    model = model_class(
                        n_estimators=100,
                        learning_rate=LEARNING_RATE,
                        max_depth=5,
                        objective="binary:logistic",
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric="logloss"
                    )
                else:
                    model = model_class(
                        n_estimators=100,
                        max_depth=5,
                        random_state=42
                    )
                
                # Calculate CV scores for common metrics
                cv_scores = {}
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    scores = cross_val_score(
                        model, X_scaled, y, cv=n_folds, scoring=metric
                    )
                    cv_scores[metric] = scores.mean()
                    cv_scores[f"{metric}_std"] = scores.std()
                
                ensemble_results[name] = cv_scores
            
            # Average results across models
            cv_results = {}
            
            for metric in ["accuracy", "precision", "recall", "f1"]:
                cv_results[metric] = np.mean([
                    results[metric] for results in ensemble_results.values()
                ])
                cv_results[f"{metric}_std"] = np.mean([
                    results[f"{metric}_std"] for results in ensemble_results.values()
                ])
                
        else:
            # Create a fresh model for CV
            if self.model_type == "xgboost":
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=LEARNING_RATE,
                    max_depth=5,
                    objective="binary:logistic",
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss"
                )
            elif self.model_type == "lightgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=LEARNING_RATE,
                    max_depth=5,
                    objective="binary",
                    random_state=42
                )
            
            # Calculate CV scores for common metrics
            for metric in ["accuracy", "precision", "recall", "f1"]:
                scores = cross_val_score(
                    model, X_scaled, y, cv=n_folds, scoring=metric
                )
                cv_results[metric] = scores.mean()
                cv_results[f"{metric}_std"] = scores.std()
        
        return cv_results
    
    def save(self) -> str:
        """
        Save the model to disk.
        
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on model type
        if self.model_type == "neural":
            # Save neural network
            torch.save(self.model.state_dict(), self.model_path.with_suffix(".pt"))
            
            # Save feature names and scaler separately
            with open(self.model_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump({
                    "feature_names": self.feature_names,
                    "scaler": self.scaler,
                    "model_type": self.model_type,
                    "input_dim": self.model.model[0].in_features if self.model else None
                }, f)
                
            logger.info(f"Saved neural network model to {self.model_path.with_suffix('.pt')}")
            return str(self.model_path.with_suffix(".pt"))
            
        elif self.model_type == "ensemble":
            # Save ensemble models
            ensemble_path = self.model_path.with_suffix(".ensemble")
            ensemble_path.mkdir(exist_ok=True)
            
            for name, model in self.model.items():
                model_path = ensemble_path / f"{name}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            
            # Save feature names and scaler
            with open(ensemble_path / "metadata.pkl", "wb") as f:
                pickle.dump({
                    "feature_names": self.feature_names,
                    "scaler": self.scaler,
                    "model_type": self.model_type
                }, f)
                
            logger.info(f"Saved ensemble models to {ensemble_path}")
            return str(ensemble_path)
            
        else:
            # Save standard model with pickle
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "scaler": self.scaler,
                    "model_type": self.model_type
                }, f)
                
            logger.info(f"Saved model to {self.model_path}")
            return str(self.model_path)
    
    def load(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            bool: True if successful
        """
        # Check model type and load accordingly
        if self.model_type == "neural":
            # Load metadata first
            meta_path = self.model_path.with_suffix(".pkl")
            if not meta_path.exists():
                logger.error(f"Neural network metadata file not found: {meta_path}")
                return False
                
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
                
            self.feature_names = metadata["feature_names"]
            self.scaler = metadata["scaler"]
            input_dim = metadata["input_dim"]
            
            # Create model
            self.model = SimpleNeuralNet(input_dim).to(self.device)
            
            # Load weights
            model_path = self.model_path.with_suffix(".pt")
            if not model_path.exists():
                logger.error(f"Neural network weights file not found: {model_path}")
                return False
                
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Loaded neural network from {model_path}")
            return True
            
        elif self.model_type == "ensemble":
            # Load ensemble models
            ensemble_path = self.model_path.with_suffix(".ensemble")
            if not ensemble_path.exists() or not ensemble_path.is_dir():
                logger.error(f"Ensemble directory not found: {ensemble_path}")
                return False
                
            # Load metadata
            meta_path = ensemble_path / "metadata.pkl"
            if not meta_path.exists():
                logger.error(f"Ensemble metadata file not found: {meta_path}")
                return False
                
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
                
            self.feature_names = metadata["feature_names"]
            self.scaler = metadata["scaler"]
            
            # Load models
            self.model = {}
            
            for name in ["xgboost", "random_forest"]:
                model_path = ensemble_path / f"{name}.pkl"
                if not model_path.exists():
                    logger.error(f"Ensemble model file not found: {model_path}")
                    return False
                    
                with open(model_path, "rb") as f:
                    self.model[name] = pickle.load(f)
            
            logger.info(f"Loaded ensemble models from {ensemble_path}")
            return True
            
        else:
            # Load standard model
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.scaler = data["scaler"]
            
            logger.info(f"Loaded model from {self.model_path}")
            return True
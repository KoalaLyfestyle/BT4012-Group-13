import os
import json
import pickle
import joblib
import pandas as pd
import requests
from pathlib import Path
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import TruncatedSVD
import sys

# Define SafeTruncatedSVD class as it is needed for unpickling
class SafeTruncatedSVD(TransformerMixin, BaseEstimator):
    """
    Adapter for TruncatedSVD that ensures n_components <= n_features at fit time.
    """
    def __init__(self, n_components=100, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self._svd = None

    def fit(self, X, y=None):
        n_features = X.shape[1]
        safe_n = max(1, min(self.n_components, n_features))
        self._svd = TruncatedSVD(n_components=safe_n, random_state=self.random_state)
        self._svd.fit(X)
        self.n_components_ = safe_n
        return self

    def transform(self, X):
        return self._svd.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Inject into __main__ so pickle can find it if it was saved as __main__.SafeTruncatedSVD
# This is a common issue when pickling classes defined in notebooks.
if '__main__' in sys.modules:
    sys.modules['__main__'].SafeTruncatedSVD = SafeTruncatedSVD

def get_experiments_dir():
    # Assuming App is at root/App and experiments is at root/experiments
    # We are running from App/ usually, but let's be robust
    current_dir = Path(os.getcwd())
    if (current_dir / "experiments").exists():
        return current_dir / "experiments"
    elif (current_dir.parent / "experiments").exists():
        return current_dir.parent / "experiments"
    else:
        # Fallback to absolute path if known or user provided
        return Path("/Users/winston/Documents/School/Y3S1/BT4012/Group Project/experiments")

def get_available_models():
    exp_dir = get_experiments_dir()
    models = []
    if not exp_dir.exists():
        return []

    for d in exp_dir.iterdir():
        if d.is_dir():
            metrics_path = d / "metrics.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    models.append({
                        "id": d.name,
                        "name": metrics.get("experiment_name", d.name),
                        "type": metrics.get("model_type", "Unknown"),
                        "vectorizer": metrics.get("vectorizer", "Unknown"),
                        "auc": metrics.get("average_metrics", {}).get("roc_auc", 0),
                        "path": d,
                        "best_fold": metrics.get("best_fold", 1),
                        "fold_models": metrics.get("fold_models", [])
                    })
                except Exception as e:
                    print(f"Error reading metrics for {d.name}: {e}")
    
    # Sort by AUC descending
    models.sort(key=lambda x: x["auc"], reverse=True)
    return models

def load_model(experiment_id, fold=None):
    exp_dir = get_experiments_dir() / experiment_id
    metrics_path = exp_dir / "metrics.json"
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    if fold is None:
        fold = metrics.get("best_fold", 1)
    
    # Find model info for this fold
    fold_info = next((f for f in metrics.get("fold_models", []) if f["fold"] == fold), None)
    if not fold_info:
        raise ValueError(f"Fold {fold} not found for experiment {experiment_id}")
    
    pipeline_path = fold_info.get("pipeline_path")
    model_path = fold_info.get("model_path")
    
    if pipeline_path:
        # Pickle or Joblib
        full_path = exp_dir / pipeline_path
        if pipeline_path.endswith(".pkl"):
            with open(full_path, "rb") as f:
                model = pickle.load(f)
        elif pipeline_path.endswith(".joblib"):
            model = joblib.load(full_path)
        else:
            raise ValueError(f"Unknown pipeline format: {pipeline_path}")
        return model
    
    elif model_path:
        # Keras or Transformers
        full_path = exp_dir / model_path
        if model_path.endswith(".keras"):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(full_path)
                return model
            except ImportError:
                raise ImportError("TensorFlow not installed")
        else:
            # Assume transformers directory
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model = AutoModelForSequenceClassification.from_pretrained(full_path)
                tokenizer = AutoTokenizer.from_pretrained(full_path)
                return (model, tokenizer)
            except ImportError:
                raise ImportError("Transformers not installed")
    
    raise ValueError("No model path found in metrics")

def fetch_phishing_urls(limit=50):
    """
    Fetch recent phishing URLs from OpenPhish (free feed).
    """
    url = "https://openphish.com/feed.txt"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            urls = response.text.strip().split('\n')
            return urls[:limit]
    except Exception as e:
        print(f"Error fetching phishing URLs: {e}")
    return []

def get_feature_columns(experiment_id):
    """
    Get the list of feature names expected by the model.
    """
    exp_dir = get_experiments_dir() / experiment_id
    features_path = exp_dir / "feature_names_all_folds.json"
    
    if features_path.exists():
        with open(features_path, "r") as f:
            data = json.load(f)
            # Return features for fold 1 (assuming all folds have same features)
            if data:
                return data[0].get("features", [])
    return []

def check_gpu_status():
    """
    Check for GPU availability (CUDA, MPS, or TF GPU).
    Returns a dictionary with status and details.
    """
    status = {
        "available": False,
        "device": "CPU",
        "details": []
    }
    
    # Check PyTorch
    try:
        print("Checking PyTorch availability...")
        import torch
        if torch.cuda.is_available():
            status["available"] = True
            status["device"] = "CUDA (NVIDIA)"
            status["details"].append(f"Torch: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            status["available"] = True
            status["device"] = "MPS (Apple Silicon)"
            status["details"].append("Torch: MPS Available")
    except ImportError:
        status["details"].append("Torch: Not Installed")

    # Check TensorFlow
    try:
        print("Checking TensorFlow availability...")
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            status["available"] = True
            if status["device"] == "CPU":
                status["device"] = "TF GPU"
            status["details"].append(f"TF: {len(gpus)} GPU(s)")
    except ImportError:
        status["details"].append("TF: Not Installed")
        
    return status

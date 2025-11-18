# Model Tracking and Evaluation Guide

## Overview

This document outlines the standard metrics, artifacts, and organization structure for tracking all models in this competition.

---

## Directory Structure

```
experiments/
├── MODEL_TRACKING.md (this file)
├── exp_1_tfidf_log/
│   ├── pipeline_fold_1.pkl (contains vectorizer + model)
│   ├── pipeline_fold_2.pkl
│   ├── pipeline_fold_3.pkl
│   ├── pipeline_fold_4.pkl
│   ├── pipeline_fold_5.pkl
│   ├── feature_names_all_folds.json
│   ├── metrics.json
│   └── exp_1_tfidf_log_prediction.csv
├── exp_2_[name]/
└── ...
```

---

## Required Artifacts Per Experiment

### 1. Pipeline Files (Recommended Approach)

- **For Cross-Validation**: Save each fold's complete pipeline
  - `pipeline_fold_1.pkl`, `pipeline_fold_2.pkl`, etc.
  - Each pipeline contains: vectorizer + model (no data leakage!)
  - Allows individual fold evaluation and ensemble approaches
  - **IMPORTANT**: Fit vectorizer separately per fold to prevent data leakage
- **Alternative (Not Recommended)**: Separate model and vectorizer files
  - Only use if you have a specific reason to fit vectorizer on all data

### 2. Feature Names

- **File**: `feature_names_all_folds.json` (if using pipelines)
- **Format**: JSON with features from each fold
  ```json
  [
    {"fold": 1, "n_features": 9523, "features": ["word1", "word2", ...]},
    {"fold": 2, "n_features": 9518, "features": ["word1", "word2", ...]},
    ...
  ]
  ```
- Enables feature importance analysis per fold

### 4. Metrics File

- **File**: `metrics.json`
- **Required Fields**:
  ```json
  {
    "experiment_name": "exp_X_description",
    "model_type": "Logistic Regression | XGBoost | Neural Network | etc.",
    "vectorizer": "TF-IDF | Word2Vec | BERT | etc.",
    "vectorizer_params": {"max_features": 10000, "ngram_range": [1, 2]},
    "model_params": {"max_iter": 1000, "solver": "liblinear"},
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "n_folds": 5,
    "best_fold": 3,
    "average_metrics": {
      "accuracy": 0.85,
      "accuracy_std": 0.02,
      "precision": 0.84,
      "precision_std": 0.03,
      "recall": 0.86,
      "recall_std": 0.02,
      "f1": 0.85,
      "f1_std": 0.02,
      "roc_auc": 0.92,
      "roc_auc_std": 0.01,
      "log_loss": 0.35,
      "log_loss_std": 0.03
    },
    "fold_metrics": [
      {
        "fold": 1,
        "accuracy": 0.83,
        "precision": 0.82,
        "recall": 0.84,
        "f1": 0.83,
        "roc_auc": 0.91,
        "log_loss": 0.37,
        "train_size": 3769,
        "val_size": 943
      }
      // ... metrics for all folds
    ],
    "fold_models": [
      {
        "fold": 1,
        "pipeline_path": "pipeline_fold_1.pkl",
        "n_features": 9523,
        "metrics": {...}
      }
      // ... info for all folds
    ],
    "optuna_study": {
      // Only for Optuna-tuned models
      "n_trials": 100,
      "best_params": {},
      "best_value": 0.92,
      "study_path": "optuna_study.pkl"
    }
  }
  ```

### 5. Prediction File

- **File**: `{experiment_name}_prediction.csv`
- **Format**:
  ```csv
  id,probability
  0,0.8234
  1,0.1456
  ...
  ```
- Use averaged predictions from all folds for ensemble approach

---

## Pipeline-Based Approach (Recommended)

### Why Use Pipelines?

1. **Prevents Data Leakage**: Vectorizer fits only on training data per fold
2. **Clean Code**: One object contains entire preprocessing + model
3. **Easy Deployment**: Load one file, make predictions
4. **Reproducible**: Same transformations guaranteed
5. **Swappable Components**: Easy to change vectorizer or model

### Creating a Pipeline with ModelSaver (Recommended)

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from save_model import ModelSaver

def create_pipeline(vectorizer_params=None, model_params=None):
    """Create a reusable pipeline"""
    if vectorizer_params is None:
        vectorizer_params = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }

    if model_params is None:
        model_params = {
            'max_iter': 1000,
            'random_state': 42
        }

    return Pipeline([
        ('vectorizer', TfidfVectorizer(**vectorizer_params)),
        ('model', LogisticRegression(**model_params))
    ])

# Initialize saver before training
saver = ModelSaver(base_path="experiments")
saver.start_experiment(
    experiment_name="exp_1_tfidf_log",
    model_type="Logistic Regression",
    vectorizer="TF-IDF",
    vectorizer_params={'max_features': 10000, 'ngram_range': (1, 2)},
    model_params={'max_iter': 1000, 'random_state': 42},
    n_folds=5,
    save_format="pickle"
)

# Use in CV loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y), start=1):
    # Create fresh pipeline per fold
    pipeline = create_pipeline(vectorizer_params, model_params)

    # Fit on train fold only
    pipeline.fit(X_text[train_idx], y[train_idx])

    # Predict on validation
    val_proba = pipeline.predict_proba(X_text[val_idx])[:, 1]

    # Predict on test
    test_proba = pipeline.predict_proba(X_test_text)[:, 1]

    # Calculate metrics
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    fold_metric = {
        "fold": fold,
        "accuracy": accuracy_score(y[val_idx], val_proba > 0.5),
        "roc_auc": roc_auc_score(y[val_idx], val_proba),
        "log_loss": log_loss(y[val_idx], val_proba),
        "train_size": len(train_idx),
        "val_size": len(val_idx)
    }

    # Save pipeline immediately (incremental mode)
    saver.add_fold(
        fold_model=pipeline,
        fold_metric=fold_metric,
        test_predictions=test_proba
    )

# Finalize experiment
saver.finalize_experiment()
```

### Swapping Models Example

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# XGBoost pipeline
xgb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(**vectorizer_params)),
    ('model', XGBClassifier(**xgb_params))
])

# LightGBM pipeline
lgb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(**vectorizer_params)),
    ('model', LGBMClassifier(**lgb_params))
])
```

---

## Standard Metrics to Track

### Classification Metrics

1. **Accuracy**: Overall correctness
2. **Precision**: Positive predictive value
3. **Recall (Sensitivity)**: True positive rate
4. **F1 Score**: Harmonic mean of precision and recall
5. **ROC AUC**: Area under ROC curve (primary metric for ranking)
6. **Log Loss**: Probabilistic prediction quality

### Additional Metrics (Optional)

- **Specificity**: True negative rate
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced datasets
- **Average Precision Score**: Area under precision-recall curve

---

## Cross-Validation Best Practices

### Standard Setup

- **Method**: Stratified K-Fold (preserves class distribution)
- **K**: 5 folds (standard)
- **Random State**: 42 (reproducibility)

### Model Selection

- **Primary Metric**: ROC AUC (best for probability predictions)
- **Identify Best Fold**: Highest validation ROC AUC
- **Ensemble**: Average predictions from all folds for submission

### What to Save

- All fold models (enables later ensemble or analysis)
- Per-fold metrics (understanding variance)
- Best fold indicator (for single model deployment)

---

## Hyperparameter Tuning (Optuna)

### Additional Artifacts for Optuna Models

1. **Study Object**: `optuna_study.pkl`

   - Complete study with all trials
   - Enables continuation and analysis

2. **Study Visualization**: `optuna_plots/`

   - Optimization history
   - Parameter importance
   - Parallel coordinates

3. **Metrics Addition**:
   ```json
   "optuna_study": {
     "n_trials": 100,
     "best_params": {
       "learning_rate": 0.01,
       "max_depth": 7,
       "n_estimators": 500
     },
     "best_value": 0.9234,
     "study_path": "optuna_study.pkl",
     "optimization_time_seconds": 3600
   }
   ```

### Optuna Best Practices

- **Objective**: Maximize ROC AUC (or minimize log loss)
- **Trials**: Minimum 50, recommended 100-200
- **Pruning**: Enable for faster optimization
- **CV within Optuna**: Use same K-fold strategy

---

## Model Comparison

### Quick Comparison Script

```python
import json
import pandas as pd
from pathlib import Path

def compare_experiments(experiments_path="experiments/"):
    results = []
    for exp_dir in Path(experiments_path).iterdir():
        if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
            with open(exp_dir / "metrics.json") as f:
                data = json.load(f)
                results.append({
                    'experiment': data['experiment_name'],
                    'model_type': data['model_type'],
                    'roc_auc': data['average_metrics']['roc_auc'],
                    'roc_auc_std': data['average_metrics']['roc_auc_std'],
                    'log_loss': data['average_metrics']['log_loss'],
                    'best_fold': data['best_fold'],
                    'timestamp': data['timestamp']
                })

    df = pd.DataFrame(results)
    return df.sort_values('roc_auc', ascending=False)

# Usage
comparison = compare_experiments()
print(comparison)
```

---

## Loading Saved Models

### Example: Load Best Fold Pipeline

```python
import pickle
import json

# Load metrics to identify best fold
with open('experiments/exp_1_tfidf_log/metrics.json') as f:
    metrics = json.load(f)
    best_fold = metrics['best_fold']

# Load best pipeline (contains vectorizer + model)
with open(f'experiments/exp_1_tfidf_log/pipeline_fold_{best_fold}.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Make predictions (vectorizer transforms automatically)
predictions = pipeline.predict_proba(new_texts)[:, 1]
```

### Example: Ensemble All Folds

```python
import numpy as np

n_folds = 5
predictions = []

for fold in range(1, n_folds + 1):
    with open(f'experiments/exp_1_tfidf_log/pipeline_fold_{fold}.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    fold_pred = pipeline.predict_proba(new_texts)[:, 1]
    predictions.append(fold_pred)

# Average predictions
ensemble_pred = np.mean(predictions, axis=0)
```

### Example: Access Pipeline Components

```python
# Load pipeline
pipeline = pickle.load(open('pipeline_fold_1.pkl', 'rb'))

# Access vectorizer
vectorizer = pipeline.named_steps['vectorizer']
feature_names = vectorizer.get_feature_names_out()

# Access model
model = pipeline.named_steps['model']
coefficients = model.coef_

# Transform text manually
X_transformed = vectorizer.transform(['some text'])
```

---

## Using the ModelSaver Utility

### Incremental Mode (Recommended)

The `save_model.py` utility provides an incremental saving mode that saves models as you train them. This is the **recommended approach** for several reasons:

1. **No data loss**: If training crashes, already-trained folds are saved
2. **Progress tracking**: See metrics for each fold as it completes
3. **Memory efficient**: Models are saved to disk immediately
4. **Cleaner code**: Automatic metrics calculation and file management

#### Basic Usage

```python
from save_model import ModelSaver

# 1. Initialize before training starts
saver = ModelSaver(base_path="experiments")
saver.start_experiment(
    experiment_name="exp_1_tfidf_log",
    model_type="Logistic Regression",
    vectorizer="TF-IDF",
    vectorizer_params={"max_features": 10000, "ngram_range": (1, 2)},
    model_params={"max_iter": 1000, "solver": "liblinear"},
    n_folds=5,
    save_format="pickle"  # or "joblib", "keras", "transformers"
)

# 2. In your cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    # Train your model
    pipeline = create_pipeline()
    pipeline.fit(X[train_idx], y[train_idx])

    # Calculate metrics
    val_proba = pipeline.predict_proba(X[val_idx])[:, 1]
    fold_metric = {
        "fold": fold,
        "accuracy": accuracy_score(y[val_idx], val_proba > 0.5),
        "precision": precision_score(y[val_idx], val_proba > 0.5),
        "recall": recall_score(y[val_idx], val_proba > 0.5),
        "f1": f1_score(y[val_idx], val_proba > 0.5),
        "roc_auc": roc_auc_score(y[val_idx], val_proba),
        "log_loss": log_loss(y[val_idx], val_proba),
        "train_size": len(train_idx),
        "val_size": len(val_idx)
    }

    # Get test predictions
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    # Save immediately
    saver.add_fold(
        fold_model=pipeline,
        fold_metric=fold_metric,
        test_predictions=test_proba
    )

# 3. Finalize after all folds complete
saver.finalize_experiment()
```

#### With Optuna

```python
# After Optuna study completes
saver.finalize_experiment(
    optuna_study=study,
    optuna_params={
        "n_trials": 100,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study_path": "optuna_study.pkl"
    }
)
```

### Batch Mode (Alternative)

If you already have all models and metrics, you can save them all at once:

```python
from save_model import ModelSaver

saver = ModelSaver(base_path="experiments")
saver.save_experiment(
    experiment_name="exp_1_tfidf_log",
    model_type="Logistic Regression",
    vectorizer="TF-IDF",
    fold_models=[pipeline1, pipeline2, pipeline3, pipeline4, pipeline5],
    fold_metrics=[metrics1, metrics2, metrics3, metrics4, metrics5],
    vectorizer_params={"max_features": 10000, "ngram_range": (1, 2)},
    model_params={"max_iter": 1000, "solver": "liblinear"},
    test_predictions=test_proba_avg,
    save_format="pickle"
)
```

### Supported Model Types

- **pickle/joblib**: Standard sklearn pipelines and models (default)
- **keras**: Neural networks saved in .keras format
- **transformers**: HuggingFace models with tokenizers

```python
# For Neural Networks
saver.start_experiment(..., save_format="keras")

# For Transformers
saver.start_experiment(..., save_format="transformers")
# Then pass (model, tokenizer) tuple to add_fold()
```

---

## Naming Conventions

### Experiment Names

- **Format**: `exp_{number}_{descriptor}`
- **Examples**:
  - `exp_1_tfidf_log` - TF-IDF + Logistic Regression
  - `exp_2_tfidf_xgb` - TF-IDF + XGBoost
  - `exp_3_bert_nn` - BERT embeddings + Neural Network
  - `exp_4_ensemble_meta` - Meta-learner ensemble

### Model Types

- Logistic Regression
- Ridge Classifier
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Neural Network
- BERT Fine-tuned
- Ensemble

### Vectorizer Types

- TF-IDF
- Count Vectorizer
- Word2Vec
- GloVe
- FastText
- BERT
- RoBERTa
- Custom

---

## Checklist Before Moving to Next Model

- [ ] All fold pipelines saved (or models if not using pipelines)
- [ ] Feature names saved (per fold if using pipelines)
- [ ] metrics.json created with all required fields
- [ ] Vectorizer and model parameters documented
- [ ] Prediction CSV generated
- [ ] Best fold identified
- [ ] Average metrics calculated
- [ ] **No data leakage**: Confirmed vectorizer fits per fold, not on all data
- [ ] Experiment documented in this file (if needed)
- [ ] Code in notebook is clean and reproducible
- [ ] Used `save_model.py` utility for consistent artifact management

---

## Notes

- Always use `random_state=42` for reproducibility
- Save models using `pickle` (or `joblib` for large models)
- Keep raw predictions for ensemble methods
- Document any custom preprocessing steps
- Track computational time for complex models
- Version control: commit after each successful experiment

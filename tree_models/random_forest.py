# === Data Info ===
# Submission shape: (70739, 2)
# Test shape: (70739, 56)

# Submission columns: ['index', 'label']
# Submission head:
#    index     label
# 0      0  0.000011
# 1      1  0.002539
# 2      2  0.999884
# 3      3  1.000000
# 4      4  1.000000

# ==================================================
# === SUBMISSION PERFORMANCE ON TEST SET ===
# ==================================================
# Accuracy: 0.9999717270529693
# ROC-AUC:  1.0

# === Confusion Matrix ===
# [[40455     0]
#  [    2 30282]]

# True Negatives:  40,455
# False Positives: 0
# False Negatives: 2
# True Positives:  30,282

# === Classification Report ===
#               precision    recall  f1-score   support

#   Legitimate       1.00      1.00      1.00     40455
#     Phishing       1.00      1.00      1.00     30284

#     accuracy                           1.00     70739
#    macro avg       1.00      1.00      1.00     70739
# weighted avg       1.00      1.00      1.00     70739


# === Prediction Distribution ===
# Predicted Legitimate (0): 40,457 (57.19%)
# Predicted Phishing (1):   30,282 (42.81%)

# Actual Legitimate (0):    40,455 (57.19%)
# Actual Phishing (1):      30,284 (42.81%)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier


def _fill_text_series(X):
    """Convert input array-like to a 1D numpy array of strings with NaNs filled."""
    s = pd.Series(np.asarray(X).ravel()).fillna('').astype(str).values
    return s


# ========= Feature Engineering Functions ========= #

def apply_feature_engineering(X, loc_groups=None, reference_date=None):
    return X, loc_groups


def map_loc_bin(df, loc_groups=None, min_samples=30, bins=None):
    return df, loc_groups


def cap_and_log(X):
    X = X.copy()
    log_cols = []
    for col in log_cols:
        if col in X.columns:
            X[col] = np.where(X[col] > 0, np.log1p(X[col]), 0)
    return X


def drop_columns(X):
    X = X.copy()
    drop_cols = []
    return X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")


# ========= Custom Transformer to Avoid Data Leakage ========= #

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that applies feature engineering.
    
    KEY: On fit(X, y) it computes location groups using the training fold's target.
    This prevents data leakage because each CV fold will compute its own loc_groups
    from its training split only.
    
    On transform(X) it applies feature engineering using the stored loc_groups.
    """
    def __init__(self, reference_date=None, min_samples=30):
        self.reference_date = reference_date
        self.min_samples = min_samples
        self.loc_groups_ = None

    def fit(self, X, y=None):
        """
        Fit on training data. Computes loc_groups from X and y.
        """
        if y is None:
            # No target provided, create empty loc_groups (will map all to 'other')
            # This happens during pipeline building when we discover schema
            self.loc_groups_ = {
                'mapping': {},
                'low_freq': [],
                'groups': {},
                'min_samples': self.min_samples,
                'bins': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            }
            return self
        
        # Attach target temporarily to compute location mappings
        df = X.copy()
        df['label'] = np.asarray(y).ravel()
        
        # Compute loc_groups from this training fold only
        _, loc_groups = apply_feature_engineering(df, loc_groups=None, reference_date=self.reference_date)
        self.loc_groups_ = loc_groups
        return self

    def transform(self, X):
        """
        Transform data using the learned loc_groups (from fit).
        """
        X_fe, _ = apply_feature_engineering(X.copy(), loc_groups=self.loc_groups_, 
                                           reference_date=self.reference_date)
        # Drop label if it exists (from fit phase)
        if 'label' in X_fe.columns:
            X_fe = X_fe.drop(columns=['label'])
        return X_fe


# ========= Build Pipeline ========= #

def build_pipeline(X_sample):
    """
    Builds full preprocessing + model pipeline.
    
    Args:
        X_sample: A small sample of raw data (before feature engineering) to
                 determine column types after feature engineering
    """
    # Create feature engineering transformer
    feat_eng = FeatureEngineerTransformer()
    
    # Apply to sample to discover column schema (use dummy y if label exists in sample)
    sample = X_sample.head(100).copy()
    y_dummy = sample['label'] if 'label' in sample.columns else None
    if 'label' in sample.columns:
        sample = sample.drop(columns=['label'])
    X_sample_fe = feat_eng.fit_transform(sample, y_dummy)
    X_temp = drop_columns(cap_and_log(X_sample_fe))
    
    numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_temp.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(max_categories=40, handle_unknown='infrequent_if_exist'))
    ])
    
    transformers = [
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]

    # can do for textual variables
    # # TF-IDF + SVD for description
    # if 'description' in X_temp.columns:
    #     transformers.append((
    #         "desc_text",
    #         Pipeline([
    #             ("fill", FunctionTransformer(_fill_text_series, validate=False)),
    #             ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')),
    #             ("svd", TruncatedSVD(n_components=40, random_state=42))
    #         ]),
    #         'description'
    #     ))

    # # TF-IDF + SVD for screen_name
    # if 'screen_name' in X_temp.columns:
    #     transformers.append((
    #         "sn_text",
    #         Pipeline([
    #             ("fill", FunctionTransformer(_fill_text_series, validate=False)),
    #             ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1,2))),
    #             ("svd", TruncatedSVD(n_components=8, random_state=42))
    #         ]),
    #         'screen_name'
    #     ))

    preprocessor = ColumnTransformer(transformers)

    # Optuna-tuned parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    # Complete pipeline with feature engineering INSIDE
    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineerTransformer()),  # ✅ FIX: Inside pipeline!
        ("cap_log", FunctionTransformer(cap_and_log, validate=False)),
        ("drop", FunctionTransformer(drop_columns, validate=False)),
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# ========= Main ========= #

if __name__ == "__main__":

    # Load data
    train = pd.read_csv("/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/dataset/train.csv")
    test = pd.read_csv("/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/dataset/test.csv")

    # Prepare training data (NO feature engineering here!)
    X = train.drop(columns=["label"])
    y = train["label"]

    # Build pipeline (pass raw X for schema discovery)
    print("🔨 Building pipeline...")
    pipeline = build_pipeline(X)

    # Cross-validation (pipeline handles feature engineering internally)
    print("\n🔄 Running 5-fold stratified cross-validation...")
    print("   (Feature engineering computed separately for each fold)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", error_score='raise')
    
    print(f"\n📊 CV Results:")
    print(f"   Mean AUC: {auc_scores.mean():.6f}")
    print(f"   Std AUC:  {auc_scores.std():.6f}")
    print(f"   Individual folds: {[f'{s:.6f}' for s in auc_scores]}")

    # Train full model
    print("\n🎯 Training final model on full training set...")
    pipeline.fit(X, y)
    
    # Save model
    # model_path = "/Users/maximus/Downloads/SCHOOL/bt4012/bt-4012-in-class-kaggle-competiton-1-2025/models/xgboost_pipeline_fixed.pkl"
    # joblib.dump(pipeline, model_path)
    # print(f"✅ Model saved to: {model_path}")

    # Predict on test (pipeline handles feature engineering)
    print("\n🔮 Generating test predictions...")
    test_probs = pipeline.predict_proba(test)[:, 1]

    submission = pd.DataFrame({
        "index": test.index,
        "label": test_probs
    })

    submission_path = "/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")
    

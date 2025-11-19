# LightGBM Implementation
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
train = pd.read_csv("/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/dataset/train.csv")
test = pd.read_csv("/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/dataset/test.csv")

# Separate features and target
X_train = train.drop(columns=['label'])
y_train = train['label']

X_test = test.drop(columns=['label'])
y_test = test['label']

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("=== Data Overview ===")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Categorical column names: {categorical_cols}")

# Build preprocessing pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(max_categories=40, handle_unknown='infrequent_if_exist'))
])

# Combine transformers
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Create full pipeline with LightGBM
lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ))
])

# Cross-validation on training data
print("\n=== Cross-Validation ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = cross_val_score(lgbm_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"CV AUC Scores: {auc_scores}")
print(f"Mean CV AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")

# Train on full training set
print("\n=== Training LightGBM on Full Training Set ===")
lgbm_pipeline.fit(X_train, y_train)
print("Training completed!")

# Evaluate on training set
y_train_pred = lgbm_pipeline.predict(X_train)
y_train_proba = lgbm_pipeline.predict_proba(X_train)[:, 1]
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Training AUC: {train_auc:.4f}")

# Evaluate on test set
y_test_pred = lgbm_pipeline.predict(X_test)
y_test_proba = lgbm_pipeline.predict_proba(X_test)[:, 1]
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"\n=== Test Set Performance ===")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

print("\n=== Confusion Matrix (Test Set) ===")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives:  {cm[1,1]:,}")

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_test_pred, target_names=['Legitimate', 'Phishing']))

# Feature importance (from the trained LightGBM)
print("\n=== Top 20 Feature Importances ===")
# Get feature names after preprocessing
feature_names = (numeric_cols + 
                 list(lgbm_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .named_steps['encoder']
                      .get_feature_names_out(categorical_cols)))

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': lgbm_pipeline.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(20))

# Save predictions to submission file
submission_lgbm = pd.DataFrame({
    "index": test.index,
    "label": y_test_proba
})

submission_path = "/Users/maximus/Downloads/SCHOOL/bt4012/BT4012-Group-13/submission_lgbm.csv"
submission_lgbm.to_csv(submission_path, index=False)
print(f"\n✅ LightGBM submission saved to: {submission_path}")

# Store pipeline for later comparison
lgbm_model = lgbm_pipeline
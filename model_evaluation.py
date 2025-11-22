import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, log_loss
from io import StringIO

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Define experiment groups
EXPERIMENTS = {
    "Linear": [
        "exp_1_numeric_lr",
        "exp_1_tfidf_lr",
        "exp_1_combined_lr",
        "exp_1_combined_svm",
        "exp_1_combined_linear_svc_optuna"
    ],
    "Tree": [
        "exp_2_random_forest_numeric",
        "exp_2_random_forest_all",
        "exp_2_xgboost_all",
        "exp_2_lgbm_all",
        "exp_2_catboost_all",
        "exp_2_catboost_optuna"
    ],
    "Neural Network": [
        "exp_3_mlp_baseline",
        "exp_3_charcnn",
        "exp_3_bilstm",
        "exp_3_hybrid",
    ],
    "Transformer": [
        "exp_4_deberta_url_only",
        "exp_4_deberta_url_features"
    ]
}

BASE_PATH = Path("experiments")
TEST_DATA_PATH = Path("dataset/df_test_feature_engineered.csv")
ARTIFACTS_PATH = Path("artifacts")

# Create artifacts directory if it doesn't exist
ARTIFACTS_PATH.mkdir(exist_ok=True)

def load_metrics(exp_name):
    metrics_path = BASE_PATH / exp_name / "metrics.json"
    if not metrics_path.exists():
        print(f"Warning: Metrics file not found for {exp_name}")
        return None
    
    with open(metrics_path, "r") as f:
        return json.load(f)

def get_test_labels():
    if not TEST_DATA_PATH.exists():
        print(f"Warning: Test data not found at {TEST_DATA_PATH}")
        return None
    
    try:
        df = pd.read_csv(TEST_DATA_PATH)
        if "target" in df.columns:
            return df["target"].values
        elif "label" in df.columns:
            return df["label"].values
        else:
            print("Error: Could not find target/label column in test data")
            return None
    except Exception as e:
        print(f"Error reading test data: {e}")
        return None

def evaluate_test_set(exp_name, y_true):
    pred_path = BASE_PATH / exp_name / f"{exp_name}_prediction.csv"
    if not pred_path.exists():
        print(f"Warning: Prediction file not found for {exp_name}")
        return None
    
    try:
        preds_df = pd.read_csv(pred_path)
        
        # Extract probabilities
        if "probability" in preds_df.columns:
            y_scores = preds_df["probability"].astype(float).values
        elif "proba" in preds_df.columns:
            y_scores = preds_df["proba"].astype(float).values
        elif preds_df.shape[1] >= 2:
            y_scores = preds_df.iloc[:, -1].astype(float).values
        else:
            print(f"Error: Could not locate probability column for {exp_name}")
            return None
            
        # Ensure lengths match
        if len(y_scores) != len(y_true):
            # Try to align if IDs are present
            if "id" in preds_df.columns:
                 # Assuming test data is indexed 0 to N-1
                 # This is risky without ID in test data, but let's assume standard order if lengths mismatch
                 # Actually, if lengths mismatch, we can't reliably evaluate unless we have IDs in both
                 # Let's just truncate to the shorter one if it's a small difference, or fail
                 min_len = min(len(y_scores), len(y_true))
                 if abs(len(y_scores) - len(y_true)) > 100: # Arbitrary threshold
                     print(f"Error: Length mismatch for {exp_name}: preds={len(y_scores)}, true={len(y_true)}")
                     return None
                 y_scores = y_scores[:min_len]
                 y_true_eval = y_true[:min_len]
            else:
                 min_len = min(len(y_scores), len(y_true))
                 y_scores = y_scores[:min_len]
                 y_true_eval = y_true[:min_len]
        else:
            y_true_eval = y_true

        # Calculate metrics
        metrics = {}
        metrics["roc_auc"] = roc_auc_score(y_true_eval, y_scores)
        metrics["log_loss"] = log_loss(y_true_eval, y_scores)
        
        # Threshold metrics (0.5)
        y_pred = (y_scores > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(y_true_eval, y_pred)
        metrics["recall"] = recall_score(y_true_eval, y_pred)
        metrics["precision"] = precision_score(y_true_eval, y_pred)
        metrics["f1"] = f1_score(y_true_eval, y_pred)
        
        return metrics

    except Exception as e:
        print(f"Error evaluating {exp_name}: {e}")
        return None

def create_metrics_table():
    print("\n=== Creating Metrics Table ===")
    y_true = get_test_labels()
    if y_true is None:
        print("Skipping test metrics due to missing labels")
    
    all_metrics = []
    
    for group, exp_list in EXPERIMENTS.items():
        for exp_name in exp_list:
            data = load_metrics(exp_name)
            if not data:
                continue
                
            avg_metrics = data.get("average_metrics", {})
            
            # Get test metrics
            test_metrics = {}
            if y_true is not None:
                test_metrics = evaluate_test_set(exp_name, y_true) or {}
            
            # Basic info
            row = {
                "Group": group,
                "Experiment": exp_name,
                "Model": data.get("model_type", "N/A"),
                "Vectorizer": data.get("vectorizer", "N/A"),
                
                # CV Metrics
                "CV ROC AUC": f"{avg_metrics.get('roc_auc', 0):.4f} ± {avg_metrics.get('roc_auc_std', 0):.4f}",
                "CV Recall": f"{avg_metrics.get('recall', 0):.4f} ± {avg_metrics.get('recall_std', 0):.4f}",
                "CV F1": f"{avg_metrics.get('f1', 0):.4f} ± {avg_metrics.get('f1_std', 0):.4f}",
                
                # Test Metrics
                "Test ROC AUC": f"{test_metrics.get('roc_auc', 0):.4f}" if test_metrics else "N/A",
                "Test Recall": f"{test_metrics.get('recall', 0):.4f}" if test_metrics else "N/A",
                "Test F1": f"{test_metrics.get('f1', 0):.4f}" if test_metrics else "N/A",
                
                # Raw values for sorting/plotting later (hidden in CSV usually, but good for DF)
                "_cv_roc_auc_val": avg_metrics.get('roc_auc', 0),
                "_test_roc_auc_val": test_metrics.get('roc_auc', 0) if test_metrics else 0,
                "_cv_recall_val": avg_metrics.get('recall', 0),
                "_test_recall_val": test_metrics.get('recall', 0) if test_metrics else 0
            }
            all_metrics.append(row)
            
    df = pd.DataFrame(all_metrics)
    
    # Save to CSV and Markdown (drop internal columns)
    display_cols = [c for c in df.columns if not c.startswith("_")]
    df[display_cols].to_csv(ARTIFACTS_PATH / "model_comparison_table.csv", index=False)
    with open(ARTIFACTS_PATH / "model_comparison_table.md", "w") as f:
        f.write(df[display_cols].to_markdown(index=False))
        
    print("Metrics table saved to model_comparison_table.csv and model_comparison_table.md")
    return df

def plot_cv_roc_auc_boxplot():
    print("\n=== Creating CV ROC AUC Boxplot ===")
    plot_data = []
    
    for group, exp_list in EXPERIMENTS.items():
        for exp_name in exp_list:
            data = load_metrics(exp_name)
            if not data:
                continue
            
            fold_metrics = data.get("fold_metrics", [])
            for fold in fold_metrics:
                plot_data.append({
                    "Experiment": exp_name,
                    "Group": group,
                    "ROC AUC": fold.get("roc_auc", 0)
                })
                
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df_plot, x="Experiment", y="ROC AUC", hue="Group", dodge=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Cross-Validation ROC AUC Distribution across Folds")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "cv_roc_auc_boxplot.png")
    print(f"Boxplot saved to {ARTIFACTS_PATH / 'cv_roc_auc_boxplot.png'}")

def plot_cv_vs_test_performance(df_metrics):
    print("\n=== Creating CV vs Test Performance Plots ===")
    
    # Filter out rows without test metrics
    df_plot = df_metrics[df_metrics["Test ROC AUC"] != "N/A"].copy()
    
    if df_plot.empty:
        print("No test metrics available for plotting.")
        return

    # Prepare data for plotting (melt)
    # ROC AUC
    roc_data = []
    for _, row in df_plot.iterrows():
        roc_data.append({
            "Experiment": row["Experiment"],
            "Group": row["Group"],
            "Metric Type": "CV",
            "Score": row["_cv_roc_auc_val"]
        })
        roc_data.append({
            "Experiment": row["Experiment"],
            "Group": row["Group"],
            "Metric Type": "Test",
            "Score": row["_test_roc_auc_val"]
        })
    
    df_roc = pd.DataFrame(roc_data)
    
    # Plot ROC AUC
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df_roc, x="Experiment", y="Score", hue="Metric Type", palette="muted")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.8, 1.0) # Zoom in as scores are likely high
    plt.title("ROC AUC: Cross-Validation vs Test Set")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "cv_vs_test_roc_auc.png")
    print(f"Saved {ARTIFACTS_PATH / 'cv_vs_test_roc_auc.png'}")
    
    # Recall
    recall_data = []
    for _, row in df_plot.iterrows():
        recall_data.append({
            "Experiment": row["Experiment"],
            "Group": row["Group"],
            "Metric Type": "CV",
            "Score": row["_cv_recall_val"]
        })
        recall_data.append({
            "Experiment": row["Experiment"],
            "Group": row["Group"],
            "Metric Type": "Test",
            "Score": row["_test_recall_val"]
        })
        
    df_recall = pd.DataFrame(recall_data)
    
    # Plot Recall
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df_recall, x="Experiment", y="Score", hue="Metric Type", palette="pastel")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.7, 1.0)
    plt.title("Recall: Cross-Validation vs Test Set")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "cv_vs_test_recall.png")
    print(f"Saved {ARTIFACTS_PATH / 'cv_vs_test_recall.png'}")

def get_feature_importance(pipeline, feature_names):
    # Try to find the model step
    model = None
    if hasattr(pipeline, "named_steps"):
        if "model" in pipeline.named_steps:
            model = pipeline.named_steps["model"]
        elif "classifier" in pipeline.named_steps:
            model = pipeline.named_steps["classifier"]
        else:
            # Last step is usually the model
            model = pipeline.steps[-1][1]
    else:
        model = pipeline

    # Extract importance
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    
    if importances is None:
        return None
        
    # Match with feature names
    if len(importances) != len(feature_names):
        print(f"Warning: Feature count mismatch. Model: {len(importances)}, Names: {len(feature_names)}")
        return None
        
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

def plot_feature_importance():
    print("\n=== Creating Feature Importance Plots ===")
    
    target_models = [
        "exp_2_catboost_all",
        "exp_2_xgboost_all",
        "exp_2_lgbm_all",
        "exp_2_random_forest_all"
    ]
    
    for exp_name in target_models:
        print(f"Processing {exp_name}...")
        exp_dir = BASE_PATH / exp_name
        
        # Load feature names
        feat_file = exp_dir / "feature_names_all_folds.json"
        if not feat_file.exists():
            print(f"  Feature names file not found for {exp_name}")
            continue
            
        with open(feat_file, "r") as f:
            feat_data = json.load(f)
            
        # Load metrics to find best fold
        metrics_data = load_metrics(exp_name)
        best_fold = metrics_data.get("best_fold", 1)
        
        # Get features for best fold
        feature_names = next((f["features"] for f in feat_data if f["fold"] == best_fold), None)
        if not feature_names:
            print(f"  Could not find feature names for fold {best_fold}")
            continue
            
        # Load pipeline for best fold
        pipeline_path = exp_dir / f"pipeline_fold_{best_fold}.pkl"
        if not pipeline_path.exists():
            print(f"  Pipeline file not found: {pipeline_path}")
            continue
            
        try:
            with open(pipeline_path, "rb") as f:
                pipeline = pickle.load(f)
        except Exception as e:
            print(f"  Error loading pipeline: {e}")
            continue
            
        # Get importance
        df_imp = get_feature_importance(pipeline, feature_names)
        if df_imp is None:
            print(f"  Could not extract feature importance for {exp_name}")
            continue
            
        # Plot top 20
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_imp.head(20), x="Importance", y="Feature", palette="viridis")
        plt.title(f"Top 20 Feature Importance - {exp_name} (Fold {best_fold})")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_PATH / f"feature_importance_{exp_name}.png")
        print(f"  Saved plot to {ARTIFACTS_PATH / f'feature_importance_{exp_name}.png'}")

def plot_submission_distributions():
    print("\n=== Creating Submission Distribution Plots ===")
    
    all_predictions = []
    
    for group, exp_list in EXPERIMENTS.items():
        for exp_name in exp_list:
            pred_path = BASE_PATH / exp_name / f"{exp_name}_prediction.csv"
            if not pred_path.exists():
                print(f"Warning: Prediction file not found for {exp_name}")
                continue
            
            try:
                preds_df = pd.read_csv(pred_path)
                
                # Extract probabilities
                if "probability" in preds_df.columns:
                    y_scores = preds_df["probability"].astype(float).values
                elif "proba" in preds_df.columns:
                    y_scores = preds_df["proba"].astype(float).values
                elif preds_df.shape[1] >= 2:
                    y_scores = preds_df.iloc[:, -1].astype(float).values
                else:
                    print(f"  Could not locate probability column for {exp_name}")
                    continue
                
                for score in y_scores:
                    all_predictions.append({
                        "Experiment": exp_name,
                        "Group": group,
                        "Probability": score
                    })
                    
            except Exception as e:
                print(f"  Error reading predictions for {exp_name}: {e}")
                continue
    
    if not all_predictions:
        print("No predictions found to plot")
        return
    
    df_preds = pd.DataFrame(all_predictions)
    
    # Plot 1: Distribution by group
    plt.figure(figsize=(14, 8))
    for group in df_preds["Group"].unique():
        group_data = df_preds[df_preds["Group"] == group]
        plt.hist(group_data["Probability"], bins=50, alpha=0.5, label=group, density=True)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Submission Probability Distribution by Model Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "submission_distribution_by_group.png")
    print(f"Saved {ARTIFACTS_PATH / 'submission_distribution_by_group.png'}")
    
    # Plot 2: Individual violin plots for each experiment
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=df_preds, x="Experiment", y="Probability", hue="Group", dodge=False, cut=0)
    plt.xticks(rotation=45, ha="right")
    plt.title("Submission Probability Distribution by Experiment")
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "submission_distribution_by_experiment.png")
    print(f"Saved {ARTIFACTS_PATH / 'submission_distribution_by_experiment.png'}")
    
    # Plot 3: Box plot showing quartiles
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_preds, x="Experiment", y="Probability", hue="Group", dodge=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Submission Probability Distribution (Box Plot)")
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_PATH / "submission_distribution_boxplot.png")
    print(f"Saved {ARTIFACTS_PATH / 'submission_distribution_boxplot.png'}")

if __name__ == "__main__":
    metrics_df = create_metrics_table()
    plot_cv_roc_auc_boxplot()
    plot_cv_vs_test_performance(metrics_df)
    plot_feature_importance()
    plot_submission_distributions()

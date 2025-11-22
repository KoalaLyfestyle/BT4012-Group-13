import optuna
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt
import pickle
import joblib
import sys
from pathlib import Path

# Set style
import seaborn as sns
sns.set_theme(style="whitegrid")

ARTIFACTS_PATH = Path("artifacts")
ARTIFACTS_PATH.mkdir(exist_ok=True)

study_path = "experiments/exp_1_combined_linear_svc_optuna/optuna_study.pkl"

if not Path(study_path).exists():
    print(f"Error: File not found at {study_path}")
    sys.exit(1)

print(f"Loading study from {study_path}...")

# Try loading
study = None
try:
    with open(study_path, "rb") as f:
        study = pickle.load(f)
except Exception as e:
    print(f"Pickle load failed: {e}")
    try:
        study = joblib.load(study_path)
        print("Joblib load successful")
    except Exception as e2:
        print(f"Joblib load failed: {e2}")
        sys.exit(1)

print(f"Loaded study with {len(study.trials)} trials")

# Plot
try:
    plt.figure(figsize=(12, 8))
    # plot_param_importances returns an Axes object
    ax = vis.plot_param_importances(study)
    ax.set_title("Hyperparameter Importance - Linear SVC Optuna")
    plt.tight_layout()
    
    output_file = ARTIFACTS_PATH / "optuna_importance_exp_1_combined_linear_svc.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
except Exception as e:
    print(f"Error plotting: {e}")
    import traceback
    traceback.print_exc()

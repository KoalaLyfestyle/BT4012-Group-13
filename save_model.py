"""
Model Saving Utility for BT4012 Group Project

This module provides a comprehensive utility to save machine learning models
according to the MODEL_TRACKING.md specifications. It supports:
- Standard scikit-learn models (pickle/joblib)
- Neural Network models (.keras)
- Transformer models (HuggingFace)
- Cross-validation with multiple folds
- Optuna hyperparameter tuning tracking
- Complete metrics and artifacts management
"""

import json
import pickle
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import warnings

# Optional imports for specialized model types
try:
    import tensorflow as tf
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    warnings.warn("TensorFlow not available. Neural network saving will be disabled.")

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Transformer model saving will be disabled.")


class ModelSaver:
    """
    Comprehensive model saving utility that handles multiple model types
    and maintains experiment tracking standards.
    
    Supports two modes:
    1. Batch mode: Save all folds at once after training completes
    2. Incremental mode: Save each fold as training progresses
    """
    
    def __init__(self, base_path: str = "experiments"):
        """
        Initialize the ModelSaver.
        
        Parameters:
        -----------
        base_path : str
            Base directory for all experiments (default: "experiments")
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # State for incremental mode
        self._incremental_mode = False
        self._exp_dir = None
        self._experiment_config = {}
        self._fold_models_info = []
        self._fold_metrics = []
        self._fold_models = []
        self._test_predictions_list = []
    
    def start_experiment(
        self,
        experiment_name: str,
        model_type: str,
        vectorizer: str,
        vectorizer_params: Dict,
        model_params: Dict,
        n_folds: int,
        save_format: str = "pickle"
    ) -> Path:
        """
        Initialize an experiment for incremental saving during training.
        Call this before starting your cross-validation loop.
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment (e.g., "exp_1_tfidf_log")
        model_type : str
            Type of model (e.g., "Logistic Regression", "XGBoost")
        vectorizer : str
            Type of vectorizer/embeddings (e.g., "TF-IDF", "BERT")
        vectorizer_params : Dict
            Parameters used for the vectorizer
        model_params : Dict
            Parameters used for the model
        n_folds : int
            Number of cross-validation folds
        save_format : str
            Format for saving models: "pickle", "joblib", "keras", or "transformers"
        
        Returns:
        --------
        Path
            Path to the experiment directory
        """
        # Create experiment directory
        self._exp_dir = self.base_path / experiment_name
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configuration
        self._experiment_config = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "vectorizer": vectorizer,
            "vectorizer_params": vectorizer_params,
            "model_params": model_params,
            "n_folds": n_folds,
            "save_format": save_format,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Reset state
        self._fold_models_info = []
        self._fold_metrics = []
        self._fold_models = []
        self._test_predictions_list = []
        self._incremental_mode = True
        
        print(f"Experiment '{experiment_name}' initialized at: {self._exp_dir}")
        print(f"Mode: Incremental saving ({n_folds} folds)")
        
        return self._exp_dir
    
    def add_fold(
        self,
        fold_model: Any,
        fold_metric: Dict,
        test_predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Add a single fold's results during training.
        Call this at the end of each fold in your CV loop.
        
        Parameters:
        -----------
        fold_model : Any
            Trained model/pipeline for this fold
        fold_metric : Dict
            Metrics dictionary for this fold (must include fold number)
        test_predictions : np.ndarray, optional
            Test set predictions for this fold
        feature_names : List[str], optional
            Feature names for this fold
        """
        if not self._incremental_mode:
            raise RuntimeError("Must call start_experiment() before add_fold()")
        
        fold_num = len(self._fold_models) + 1
        
        # Ensure fold number is in metrics
        if "fold" not in fold_metric:
            fold_metric["fold"] = fold_num
        
        # Store model and metrics
        self._fold_models.append(fold_model)
        self._fold_metrics.append(fold_metric)
        
        if test_predictions is not None:
            self._test_predictions_list.append(test_predictions)
        
        # Save model immediately
        save_format = self._experiment_config["save_format"]
        model_info = self._save_single_fold_model(
            fold_model, fold_num, save_format, fold_metric, feature_names
        )
        self._fold_models_info.append(model_info)
        
        # Print progress
        roc_auc = fold_metric.get('roc_auc', 'N/A')
        if roc_auc != 'N/A':
            print(f"  Fold {fold_num}/{self._experiment_config['n_folds']} saved | ROC AUC: {roc_auc:.4f}")
        else:
            print(f"  Fold {fold_num}/{self._experiment_config['n_folds']} saved")
    
    def finalize_experiment(
        self,
        optuna_study: Optional[Any] = None,
        optuna_params: Optional[Dict] = None,
        additional_notes: Optional[str] = None
    ) -> Path:
        """
        Finalize the experiment after all folds are complete.
        Call this after your training loop finishes.
        
        Parameters:
        -----------
        optuna_study : optuna.Study, optional
            Optuna study object (if using Optuna)
        optuna_params : Dict, optional
            Optuna-specific parameters and results
        additional_notes : str, optional
            Any additional notes to include in metrics
        
        Returns:
        --------
        Path
            Path to the experiment directory
        """
        if not self._incremental_mode:
            raise RuntimeError("Must call start_experiment() before finalize_experiment()")
        
        print("\nFinalizing experiment...")
        
        # Calculate average metrics
        average_metrics = self._calculate_average_metrics(self._fold_metrics)
        
        # Identify best fold
        best_fold = self._identify_best_fold(self._fold_metrics)
        
        # Create metrics.json
        metrics_data = self._create_metrics_json(
            experiment_name=self._experiment_config["experiment_name"],
            model_type=self._experiment_config["model_type"],
            vectorizer=self._experiment_config["vectorizer"],
            vectorizer_params=self._experiment_config["vectorizer_params"],
            model_params=self._experiment_config["model_params"],
            n_folds=self._experiment_config["n_folds"],
            best_fold=best_fold,
            average_metrics=average_metrics,
            fold_metrics=self._fold_metrics,
            fold_model_info=self._fold_models_info,
            optuna_params=optuna_params
        )
        
        if additional_notes:
            metrics_data["notes"] = additional_notes
        
        with open(self._exp_dir / "metrics.json", "w") as f:
            json.dump(_make_json_serializable(metrics_data), f, indent=2, cls=NumpyJSONEncoder)
        
        # Save Optuna study if provided
        if optuna_study is not None:
            self._save_optuna_study(optuna_study, self._exp_dir)
        
        # Save averaged predictions if available
        if self._test_predictions_list:
            avg_predictions = np.mean(self._test_predictions_list, axis=0)
            self._save_prediction(avg_predictions, self._exp_dir, self._experiment_config["experiment_name"])
        
        # Print summary
        experiment_name = self._experiment_config["experiment_name"]
        print(f"\n✓ Experiment '{experiment_name}' finalized!")
        print(f"  Location: {self._exp_dir}")
        print(f"  Folds completed: {len(self._fold_models)}")
        print(f"  Best fold: {best_fold} (ROC AUC: {self._fold_metrics[best_fold-1].get('roc_auc', 'N/A'):.4f})")
        print(f"  Average ROC AUC: {average_metrics.get('roc_auc', 'N/A'):.4f} ± {average_metrics.get('roc_auc_std', 'N/A'):.4f}")
        
        # Reset state
        self._incremental_mode = False
        
        return self._exp_dir

    def _save_optuna_study(self, optuna_study: Any, exp_dir: Path):
        """Save Optuna study object."""
        study_path = exp_dir / "optuna_study.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(optuna_study, f)

        # Create plots directory
        plots_dir = exp_dir / "optuna_plots"
        plots_dir.mkdir(exist_ok=True)

        try:
            import optuna.visualization as vis

            # Save optimization history
            fig = vis.plot_optimization_history(optuna_study)
            fig.write_html(plots_dir / "optimization_history.html")

            # Save parameter importance
            fig = vis.plot_param_importances(optuna_study)
            fig.write_html(plots_dir / "param_importances.html")

            # Save parallel coordinate plot
            fig = vis.plot_parallel_coordinate(optuna_study)
            fig.write_html(plots_dir / "parallel_coordinate.html")

            print(f"  Optuna plots saved to {plots_dir}")
        except Exception as e:
            warnings.warn(f"Could not create Optuna plots: {e}")

    def _save_prediction(
        self,
        test_predictions: np.ndarray,
        exp_dir: Path,
        experiment_name: str,
    ):
        """Save prediction CSV file."""
        prediction_path = exp_dir / f"{experiment_name}_prediction.csv"

        # Create prediction dataframe
        prediction_data = {
            "id": np.arange(len(test_predictions)),
            "probability": test_predictions,
        }

        # Save as CSV
        import pandas as pd

        df = pd.DataFrame(prediction_data)
        df.to_csv(prediction_path, index=False)
        print(f"  Predictions saved to {prediction_path}")
    
    def save_experiment(
        self,
        experiment_name: str,
        model_type: str,
        vectorizer: str,
        fold_models: List[Any],
        fold_metrics: List[Dict],
        vectorizer_params: Dict,
        model_params: Dict,
        test_predictions: Optional[np.ndarray] = None,
        feature_names_per_fold: Optional[List[Dict]] = None,
        optuna_study: Optional[Any] = None,
        optuna_params: Optional[Dict] = None,
        save_format: str = "pickle"
    ) -> Path:
        """
        Save a complete experiment with all artifacts (batch mode).
        Use this when you have all fold models and metrics ready.
        For incremental saving during training, use start_experiment() -> add_fold() -> finalize_experiment().
        
        Parameters:
        -----------
        experiment_name : str
            Name of the experiment (e.g., "exp_1_tfidf_log")
        model_type : str
            Type of model (e.g., "Logistic Regression", "XGBoost", "Neural Network")
        vectorizer : str
            Type of vectorizer/embeddings (e.g., "TF-IDF", "BERT")
        fold_models : List[Any]
            List of trained models/pipelines for each fold
        fold_metrics : List[Dict]
            List of metrics dictionaries for each fold
        vectorizer_params : Dict
            Parameters used for the vectorizer
        model_params : Dict
            Parameters used for the model
        test_predictions : np.ndarray, optional
            Test set predictions (averaged across folds)
        feature_names_per_fold : List[Dict], optional
            Feature names for each fold
        optuna_study : optuna.Study, optional
            Optuna study object (if using Optuna)
        optuna_params : Dict, optional
            Optuna-specific parameters and results
        save_format : str
            Format for saving models: "pickle", "joblib", "keras", or "transformers"
        
        Returns:
        --------
        Path
            Path to the created experiment directory
        """
        # Create experiment directory
        exp_dir = self.base_path / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving experiment to: {exp_dir}")
        
        # Validate inputs
        n_folds = len(fold_models)
        if len(fold_metrics) != n_folds:
            raise ValueError(f"Number of models ({n_folds}) and metrics ({len(fold_metrics)}) must match")
        
        # Save fold models
        print(f"Saving {n_folds} fold models...")
        fold_model_info = self._save_fold_models(
            fold_models, exp_dir, save_format, fold_metrics
        )
        
        # Save feature names if provided
        if feature_names_per_fold is not None:
            print("Saving feature names...")
            self._save_feature_names(feature_names_per_fold, exp_dir)
        
        # Calculate average metrics
        print("Calculating average metrics...")
        average_metrics = self._calculate_average_metrics(fold_metrics)
        
        # Identify best fold
        best_fold = self._identify_best_fold(fold_metrics)
        
        # Create metrics.json
        print("Creating metrics.json...")
        metrics_data = self._create_metrics_json(
            experiment_name=experiment_name,
            model_type=model_type,
            vectorizer=vectorizer,
            vectorizer_params=vectorizer_params,
            model_params=model_params,
            n_folds=n_folds,
            best_fold=best_fold,
            average_metrics=average_metrics,
            fold_metrics=fold_metrics,
            fold_model_info=fold_model_info,
            optuna_params=optuna_params
        )
        
        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(_make_json_serializable(metrics_data), f, indent=2, cls=NumpyJSONEncoder)
        
        # Save Optuna study if provided
        if optuna_study is not None:
            print("Saving Optuna study...")
            self._save_optuna_study(optuna_study, exp_dir)
        
        # Save prediction file if test predictions provided
        if test_predictions is not None:
            print("Saving prediction file...")
            self._save_prediction(test_predictions, exp_dir, experiment_name)
        
        print(f"✓ Experiment '{experiment_name}' saved successfully!")
        print(f"  Location: {exp_dir}")
        print(f"  Best fold: {best_fold} (ROC AUC: {fold_metrics[best_fold-1].get('roc_auc', 'N/A'):.4f})")
        print(f"  Average ROC AUC: {average_metrics.get('roc_auc', 'N/A'):.4f} ± {average_metrics.get('roc_auc_std', 'N/A'):.4f}")
        
        return exp_dir
    
    def _save_single_fold_model(
        self,
        model: Any,
        fold_num: int,
        save_format: str,
        fold_metric: Dict,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Save a single fold's model and return metadata.
        Used in incremental mode.
        
        Returns:
        --------
        Dict
            Metadata for this fold model
        """
        model_info = {
            "fold": fold_num,
            "metrics": fold_metric
        }
        
        if save_format == "pickle":
            model_path = self._exp_dir / f"pipeline_fold_{fold_num}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            model_info["pipeline_path"] = model_path.name
            
        elif save_format == "joblib":
            model_path = self._exp_dir / f"pipeline_fold_{fold_num}.joblib"
            joblib.dump(model, model_path)
            model_info["pipeline_path"] = model_path.name
            
        elif save_format == "keras":
            if not KERAS_AVAILABLE:
                raise RuntimeError("TensorFlow is not available. Cannot save Keras models.")
            model_path = self._exp_dir / f"model_fold_{fold_num}.keras"
            model.save(model_path)
            model_info["model_path"] = model_path.name
            
        elif save_format == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers library is not available.")
            model_dir = self._exp_dir / f"transformer_fold_{fold_num}"
            model_dir.mkdir(exist_ok=True)
            
            # Save model and tokenizer
            if isinstance(model, tuple) and len(model) == 2:
                transformer_model, tokenizer = model
                transformer_model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
            else:
                model.save_pretrained(model_dir)
            
            model_info["model_path"] = model_dir.name
        
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        # 1. Handle explicit feature names
        if feature_names is not None:
            self._save_single_fold_features(fold_num, feature_names)
            model_info["n_features"] = len(feature_names)

        # 2. Try to extract from vectorizer if not provided
        try:
            if hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
                vectorizer = model.named_steps['vectorizer']
                if hasattr(vectorizer, 'get_feature_names_out'):
                    n_features = len(vectorizer.get_feature_names_out())
                    if "n_features" not in model_info:
                        model_info["n_features"] = n_features
                    
                    # Save from vectorizer if not already saved via explicit feature_names
                    if fold_num == 1 and feature_names is None:
                        self._save_single_fold_features(
                            fold_num, 
                            vectorizer.get_feature_names_out().tolist()
                        )
            elif hasattr(model, 'n_features_in_'):
                if "n_features" not in model_info:
                    model_info["n_features"] = model.n_features_in_
        except:
            pass
        
        return model_info
    
    def _save_fold_models(
        self, 
        fold_models: List[Any], 
        exp_dir: Path, 
        save_format: str,
        fold_metrics: List[Dict]
    ) -> List[Dict]:
        """
        Save models for all folds and return metadata (batch mode).
        
        Returns:
        --------
        List[Dict]
            Metadata for each fold model
        """
        fold_model_info = []
        
        for fold_idx, model in enumerate(fold_models, start=1):
            model_info = {
                "fold": fold_idx,
                "metrics": fold_metrics[fold_idx - 1]
            }
            
            if save_format == "pickle":
                model_path = exp_dir / f"pipeline_fold_{fold_idx}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                model_info["pipeline_path"] = model_path.name
                
            elif save_format == "joblib":
                model_path = exp_dir / f"pipeline_fold_{fold_idx}.joblib"
                joblib.dump(model, model_path)
                model_info["pipeline_path"] = model_path.name
                
            elif save_format == "keras":
                if not KERAS_AVAILABLE:
                    raise RuntimeError("TensorFlow is not available. Cannot save Keras models.")
                model_path = exp_dir / f"model_fold_{fold_idx}.keras"
                model.save(model_path)
                model_info["model_path"] = model_path.name
                
            elif save_format == "transformers":
                if not TRANSFORMERS_AVAILABLE:
                    raise RuntimeError("Transformers library is not available.")
                model_dir = exp_dir / f"transformer_fold_{fold_idx}"
                model_dir.mkdir(exist_ok=True)
                
                # Save model and tokenizer
                if isinstance(model, tuple) and len(model) == 2:
                    transformer_model, tokenizer = model
                    transformer_model.save_pretrained(model_dir)
                    tokenizer.save_pretrained(model_dir)
                else:
                    model.save_pretrained(model_dir)
                
                model_info["model_path"] = model_dir.name
            
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            # Try to extract number of features if available
            try:
                if hasattr(model, 'named_steps') and 'vectorizer' in model.named_steps:
                    vectorizer = model.named_steps['vectorizer']
                    if hasattr(vectorizer, 'get_feature_names_out'):
                        model_info["n_features"] = len(vectorizer.get_feature_names_out())
                elif hasattr(model, 'n_features_in_'):
                    model_info["n_features"] = model.n_features_in_
            except:
                pass
            
            fold_model_info.append(model_info)
        
        return fold_model_info
    
    def _save_single_fold_features(self, fold_num: int, feature_names: List[str]):
        """Save feature names for a single fold (incremental mode)."""
        features_file = self._exp_dir / "feature_names_all_folds.json"
        
        # Load existing data if file exists
        if features_file.exists():
            with open(features_file, "r") as f:
                feature_data = json.load(f)
        else:
            feature_data = []
        
        # Add this fold's features
        fold_info = {
            "fold": fold_num,
            "n_features": len(feature_names),
            "features": feature_names
        }
        
        feature_data.append(fold_info)
        
        # Save updated data
        with open(features_file, "w") as f:
            json.dump(_make_json_serializable(feature_data), f, indent=2, cls=NumpyJSONEncoder)
    
    def _save_feature_names(
        self, 
        feature_names_per_fold: List[Dict], 
        exp_dir: Path
    ):
        """Save feature names for all folds (batch mode)."""
        feature_data = []
        
        for fold_data in feature_names_per_fold:
            fold_info = {
                "fold": fold_data.get("fold"),
                "n_features": fold_data.get("n_features", len(fold_data.get("features", []))),
                "features": fold_data.get("features", [])
            }
            
            feature_data.append(fold_info)
        
        with open(exp_dir / "feature_names_all_folds.json", "w") as f:
            json.dump(_make_json_serializable(feature_data), f, indent=2, cls=NumpyJSONEncoder)
    
    def _calculate_average_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Calculate average and standard deviation for all metrics."""
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss']
        average_metrics = {}
        
        for key in metric_keys:
            values = [fold.get(key) for fold in fold_metrics if fold.get(key) is not None]
            if values:
                average_metrics[key] = float(np.mean(values))
                average_metrics[f"{key}_std"] = float(np.std(values))
        
        return average_metrics
    
    def _identify_best_fold(self, fold_metrics: List[Dict]) -> int:
        """Identify the best fold based on ROC AUC score."""
        roc_aucs = [fold.get('roc_auc', 0) for fold in fold_metrics]
        best_fold_idx = np.argmax(roc_aucs)
        return best_fold_idx + 1  # 1-indexed
    
    def _create_metrics_json(
        self,
        experiment_name: str,
        model_type: str,
        vectorizer: str,
        vectorizer_params: Dict,
        model_params: Dict,
        n_folds: int,
        best_fold: int,
        average_metrics: Dict,
        fold_metrics: List[Dict],
        fold_model_info: List[Dict],
        optuna_params: Optional[Dict]
    ) -> Dict:
        """Create the complete metrics.json structure."""
        metrics_data = {
            "experiment_name": experiment_name,
            "model_type": model_type,
            "vectorizer": vectorizer,
            "vectorizer_params": vectorizer_params,
            "model_params": model_params,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_folds": n_folds,
            "best_fold": best_fold,
            "average_metrics": average_metrics,
            "fold_metrics": fold_metrics,
            "fold_models": fold_model_info
        }
        
        # Add Optuna information if provided
        if optuna_params is not None:
            metrics_data["optuna_study"] = optuna_params
        
        return metrics_data


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and pandas data types gracefully.

    Converts numpy integers/floats/bools to native Python types, numpy arrays
    to lists, pandas Series/Index to Python lists and pandas Timestamps to ISO
    formatted strings. This prevents TypeError: Object of type int64 is not JSON
    serializable when writing experiment metadata.
    """
    def default(self, o):
        # Local import for optional dependencies
        try:
            import numpy as _np
        except Exception:
            _np = None

        try:
            import pandas as _pd
        except Exception:
            _pd = None

        # Numpy types
        if _np is not None:
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()

        # Pandas types
        if _pd is not None:
            if isinstance(o, _pd.Timestamp):
                return o.isoformat()
            if isinstance(o, _pd.Timedelta):
                return str(o)
            if isinstance(o, (_pd.Series, _pd.Index)):
                return o.tolist()

        # Fall back to default behaviour
        return super().default(o)


def _make_json_serializable(obj):
    """Recursively convert numpy / pandas objects to standard Python types.

    This function ensures that any nested structure handed to json.dump will only
    contain JSON-serializable Python data types (int, float, bool, str, list,
    dict). It complements the encoder by proactively converting values.
    """
    # Local imports to avoid import-time dependency errors
    try:
        import numpy as _np
    except Exception:
        _np = None

    try:
        import pandas as _pd
    except Exception:
        _pd = None

    # Recursively handle containers
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_serializable(v) for v in obj]

    # Numpy scalars and arrays
    if _np is not None:
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return _make_json_serializable(obj.tolist())

    # Pandas types
    if _pd is not None:
        if isinstance(obj, (_pd.Series, _pd.Index)):
            return _make_json_serializable(obj.tolist())
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, _pd.Timedelta):
            return str(obj)

    # Path-like objects
    try:
        from pathlib import Path as _Path
        if isinstance(obj, _Path):
            return str(obj)
    except Exception:
        pass

    # Default fallback for primitives and unknown objects
    return obj


# Convenience function for quick saving
def save_model_quick(
    experiment_name: str,
    model_type: str,
    vectorizer: str,
    fold_models: List[Any],
    fold_metrics: List[Dict],
    vectorizer_params: Dict = None,
    model_params: Dict = None,
    test_predictions: np.ndarray = None,
    save_format: str = "pickle",
    base_path: str = "experiments"
) -> Path:
    """
    Quick save function with minimal required parameters.
    
    Example:
    --------
    >>> from save_model import save_model_quick
    >>> save_model_quick(
    ...     experiment_name="exp_1_tfidf_log",
    ...     model_type="Logistic Regression",
    ...     vectorizer="TF-IDF",
    ...     fold_models=[pipeline1, pipeline2, pipeline3, pipeline4, pipeline5],
    ...     fold_metrics=[metrics1, metrics2, metrics3, metrics4, metrics5],
    ...     test_predictions=test_proba
    ... )
    """
    saver = ModelSaver(base_path=base_path)
    
    return saver.save_experiment(
        experiment_name=experiment_name,
        model_type=model_type,
        vectorizer=vectorizer,
        fold_models=fold_models,
        fold_metrics=fold_metrics,
        vectorizer_params=vectorizer_params or {},
        model_params=model_params or {},
        test_predictions=test_predictions,
        save_format=save_format
    )


# Example usage
if __name__ == "__main__":
    print("Model Saver Utility")
    print("=" * 60)
    print("\n=== INCREMENTAL MODE (Recommended for Training) ===\n")
    print("""
from save_model import ModelSaver
from sklearn.model_selection import StratifiedKFold

# Initialize saver and start experiment
saver = ModelSaver(base_path="experiments")
saver.start_experiment(
    experiment_name="exp_1_tfidf_log",
    model_type="Logistic Regression",
    vectorizer="TF-IDF",
    vectorizer_params={"max_features": 10000, "ngram_range": (1, 2)},
    model_params={"max_iter": 1000, "solver": "liblinear"},
    n_folds=5,
    save_format="pickle"
)

# Cross-validation loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    # Train your model
    pipeline = create_pipeline()
    pipeline.fit(X[train_idx], y[train_idx])
    
    # Calculate metrics
    val_proba = pipeline.predict_proba(X[val_idx])[:, 1]
    fold_metric = {
        "fold": fold,
        "accuracy": accuracy_score(y[val_idx], val_proba > 0.5),
        "roc_auc": roc_auc_score(y[val_idx], val_proba),
        # ... other metrics
    }
    
    # Get test predictions
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Save this fold immediately
    saver.add_fold(
        fold_model=pipeline,
        fold_metric=fold_metric,
        test_predictions=test_proba
    )

# Finalize after all folds complete
saver.finalize_experiment()
    """)
    
    print("\n=== BATCH MODE (All at Once) ===\n")
    print("""
from save_model import ModelSaver

# Initialize saver
saver = ModelSaver(base_path="experiments")

# Prepare your data (after training is complete)
fold_models = [pipeline1, pipeline2, pipeline3, pipeline4, pipeline5]
fold_metrics = [
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
    },
    # ... more fold metrics
]

# Save experiment
saver.save_experiment(
    experiment_name="exp_1_tfidf_log",
    model_type="Logistic Regression",
    vectorizer="TF-IDF",
    fold_models=fold_models,
    fold_metrics=fold_metrics,
    vectorizer_params={"max_features": 10000, "ngram_range": (1, 2)},
    model_params={"max_iter": 1000, "solver": "liblinear"},
    test_predictions=test_proba_avg,
    save_format="pickle"  # or "joblib", "keras", "transformers"
)

# For Optuna-tuned models, add:
# optuna_params={
#     "n_trials": 100,
#     "best_params": study.best_params,
#     "best_value": study.best_value,
#     "study_path": "optuna_study.pkl"
# }
# optuna_study=study
    """)

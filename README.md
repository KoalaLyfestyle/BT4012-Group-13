## Problem Description

Phishing is a form of online fraud where attackers deceive people into revealing sensitive information or installing malware. Unlike traditional scams, phishing includes a blend of technical deception with social engineering tactics designed to appear legitimate to users. This makes it increasingly difficult for users to distinguish between genuine websites from fraudulent ones.

The real-world impact of phishing continues to accelerate globally. Recent reports show that phishing attacks have risen to 1.96 million incidents per year, marking a 182% increase since 2021 (Einpresswire, 2025). In Singapore, phishing was the most common scam in the first half of 2025, with 3779 reported cases and losses rising by 134% year-on-year to over $30 million (Chua, 2025). Industry experts also warned that Artificial Intelligence (AI) will increase the quantity and quality of phishing scams going forward, utilising it to scale and refine their operations (Heiding et al., 2025). As such, with phishing becoming more widespread and sophisticated, addressing this issue is critical.

However, detecting phishing attacks presents several challenges. Much of the defence against phishing is through awareness campaigns and simulated phishing exercises. This approach depends on individuals recognising suspicious cues which is increasingly difficult as phishing techniques and patterns evolve over time. Additionally, many traditional rule-based phishing detection rely on filters or blacklists which can be easily bypassed once attackers learn these rules. Lastly, while large organisations may have more sophisticated security infrastructure to prevent phishing activities, the general public often lack access to such measures.

Thus, in this project, we plan to develop a phishing detection model that is effective and accessible for all individuals. By leveraging machine learning (ML) models that can detect subtle patterns not easily identifiable by users and can be continuously retrained with new data to remain adaptable, this ensures users receive proactive protection from phishing attacks.

## Dataset

We use a real world phishing dataset to train our model, published on Mendeley Data. The main details are as follows:

<table>
	<tbody>
		<tr>
			<td>Total Instances</td>
			<td>11430 URLs</td>
		</tr>
		<tr>
			<td>Class Composition</td>
			<td>5715 legitimate URLs, 5715 phishing URLs</td>
		</tr>
		<tr>
			<td>Features</td>
			<td>87 extracted numerical features, 1 `url` text feature, 1 `status` feature containing the class label.</td>
		</tr>
	</tbody>
</table>

The link to the dataset can be found [here](https://data.mendeley.com/datasets/c2gw7fy2j4/3).

## Project Folders and Key Files

### Notebooks

- `Processing.ipynb` - Data preprocessing and train-test split
- `Feature_Engineering_EDA.ipynb` - Feature engineering from URLs and exploratory data analysis
- `Feature_Engineering_LLM.ipynb` - Feature engineering using Large Language Models (LLMs)
- `exp_1_linear_models.ipynb` - Exploring Linear models (Logistic Regression, SVM) with TF-IDF and Count vectorization
- `exp_2_tree_models.ipynb` - Exploring Tree-based models (XGBoost, LightGBM, CatBoost, Random Forest)
- `exp_3_NN_models.ipynb` - Exploring Neural network architectures (MLP, CharCNN, BiLSTM, Hybrid models)
- `exp_4_transformer_models.ipynb` - Exploring Transformer-based models (DeBERTa) for URL classification
- `exp_5_ensemble_models.ipynb` - Exploring Ensemble methods combining multiple model predictions
- `Model_Evaluation.ipynb` - Model comparison and evaluation of trained models

### Datasets

- `dataset/kaggle_phishing_dataset.csv` - Raw dataset (11,430 URLs)
- `dataset/train.csv` - Training set (9,143 URLs)
- `dataset/test.csv` - Test set (2,286 URLs)
- `dataset/df_train_feature_engineered.csv` - Training set with engineered features
- `dataset/df_test_feature_engineered.csv` - Test set with engineered features

### Scripts and Utilities

- `save_model.py` - Model saving utility supporting multiple formats with cross-validation tracking

### Documentation

- `MODEL_TRACKING.md` - Experiment tracking standards and model artifact specifications
- `experiments/` - Saved models, metrics, and predictions for all experiments
- `artifacts/` - Generated visualisations and analysis outputs

_Note: DeBERTa model weights are available in the `all-models` branch using Git LFS. To download them, switch to the `all-models` branch and pull with Git LFS installed._

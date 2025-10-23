## Problem Statement

Phishing is a form of online fraud where attackers deceive people into revealing sensitive information or installing malware. Unlike traditional scams, phishing includes a blend of technical deception with social engineering, making it difficult for users to differentiate between fraudulent and authentic sites. This unique format of fraud gives us an opportunity to create a model that can detect the technical information, as well as use natural language features to decode attempts at malicious social engineering. Furthermore, vast volumes of websites are tracked through tools like Open PageRank, and labelled collections of phishing sites are maintained by many organisations, a significant portion being open source. This allows for us to test our models on larger datasets in the future, with room to experiment with multiple sources of data.

## Dataset

We use a real world phishing dataset to train our model, PhiUSIIL, published on UC Irvine’s Machine Learning Repository. The main details are as follows:

<table>
	<tbody>
		<tr>
			<td>Total Instances</td>
			<td>235795 URLs</td>
		</tr>
		<tr>
			<td>Class Composition</td>
			<td>134,850 legitimate URLs, 100,945 phishing URLs</td>
		</tr>
		<tr>
			<td>Features</td>
			<td>54 features, mix of numerical, categorical, and textual data</td>
		</tr>
	</tbody>
</table>

## Planned Methodology

In our planned methodology, we will explore two different approaches. The first approach involves utilising most of the features provided in the dataset to build a comprehensive model. However, some features, such as NoOfCSS and URLTitleMatchScore, are not easily obtainable when only the URL is available. To improve real-world applicability, our second approach restricts the feature set to those that can be directly derived from the URL itself, such as NoOfLettersInURL and URLLength. This allows users to determine whether a URL is legitimate using only the URL as input.

For this task, our main approach will be supervised learning, making use of the labelled dataset to train and evaluate models. By leveraging both existing features and engineered features, mainly via NLP techniques like TF-IDF on URL strings, we will explore a range of traditional ML models from linear models like Logistic Regression, to tree-based models such as Random Forest and XGBoost. Additionally, we also plan to explore non-linear deep learning models such as CNN and LSTM, which can directly process the raw URL string as inputs to automatically learn patterns and make predictions.

Our primary training dataset is the PHiUSIIL phishing dataset comprising 235,795 unique URLs with 54 features. For testing, we will incorporate external datasets to evaluate the generalisability of our models. This will include using Open PageRank to assess domain legitimacy and various open-source collections of phishing websites that feature more recent URLs. This lets us test whether models trained on older data like PhiUSIIL still perform well on newer threats, helping us evaluate model accuracy.

Although our dataset is relatively balanced, the real world distribution is not. Therefore, models will be evaluated with ROC AUC, which is threshold-independent and robust to class imbalance. Additionally, Recall is important as missing a real phishing site is costlier than flagging a benign site, while Accuracy may be less helpful in the case of imbalanced data. Finally, F1-score will also help us to balance between Precision and Recall if our model ends up flagging too many benign sites as phishing, as it may be costly to business owners.

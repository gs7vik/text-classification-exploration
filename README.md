Text Classification- Naive Bayes, Logistic Regression, Linear SVC, Ensemble Learning 

This repo outlines the methodology, findings, and final model selection for a text classification task based on the provided Jupyter Notebook. The objective was to build a robust model to classify text into subjects using a high-performance approach that included advanced data preprocessing, diverse feature engineering, and multiple ensemble modeling strategies.

***

### Data Cleaning and Preprocessing

The preprocessing strategy, termed involves several key steps to prepare the text data for modeling.

* **Lowercase Conversion**: The text is converted to lowercase to ensure consistency.
* **Number Handling**: Specific numerical patterns are tokenized. Years (e.g., `1961`) are replaced with `YEAR`, decimals (e.g., `50.5`) with `DECIMAL`, and other numbers with `NUM`. This helps the model recognize the semantic importance of numerical data.
* **Special Character Removal**: Most special characters are removed, but punctuation like periods, exclamation points, and question marks are retained as they can provide meaningful context.
* **Whitespace Cleanup**: Redundant whitespace is removed to standardize the text format.

***

### Approaches Used

The  exploration involved variety of machine learning approaches, categorized into individual models and ensemble strategies. The models were evaluated using 3-fold stratified cross-validation with the **$F1\_macro$ score** as the primary metric, multi-class problems with potential class imbalance.

#### **Individual Models**

The individual models, built as `Pipelines`, combine a feature extractor (TfidfVectorizer) with a classifier:

* **Multinomial Naive Bayes (MNB)**: Two variations were used with different feature sets and low `alpha` values.
* **Support Vector Machine (SVM)**: Two `LinearSVC` models were configured with different feature combinations and regularization parameters.
* **Logistic Regression (LR)**: Two `LogisticRegression` models were trained using different feature sets.
* **Random Forest (RF)**: A `RandomForestClassifier` was included for its ability to handle non-linear relationships, using a smaller feature set to manage computational load.

#### **Ensemble Strategies**

Three ensemble strategies were implemented to combine the strengths of the individual models:

1.  **Stacked Ensemble**: This meta-learning approach uses a `LogisticRegression` final estimator to combine the predictions of base models like MNB, SVM, and LR.
2.  **Hard Voting Ensemble**: This method uses a majority vote from a diverse set of models (MNB, SVM, LR, RF) to determine the final prediction.
3.  **Soft Voting Ensemble**: This approach averages the predicted probabilities from models that support them (MNB, LR, RF) to make the final prediction.

***

### Best Model and Accuracy Metrics

The cross-validation results show that the **Stacked_Ensemble** model achieved the highest performance.

| Approach | Mean F1-macro Score (%) | Standard Deviation (%) |
| :--- | :--- | :--- |
| **Stacked Ensemble** | **88.7833** | 1.4206 |
| Soft Voting | 88.2930 | 0.9658 |
| Hard Voting | 87.8908 | 1.3267 |
| Ultra_SVM_1 | 87.6898 | 1.2101 |
| Ultra_NB_1 | 87.2149 | 1.4652 |
| Ultra_NB_2 | 87.1269 | 1.3946 |
| Ultra_LR | 84.2396 | 0.7592 |

The **Stacked_Ensemble** model was selected as the best approach with a mean $F1\_macro$ score of **88.7833%**. This ensemble strategy leverages the strengths of multiple models to produce a more robust and accurate classification. The notebook then proceeds to train this model on the entire training dataset and uses it to generate predictions for the test set. The final submission file is named `submission.csv`.

Also tried a pre trained embedding model(facebook/bart-large-mnli) which excels at zero shot classification to check out the accuracy. It was quite low (macro averaged F1 score of 54.23%). It might be due to the fact that no data pre processing was done. But I am not sure if even after doing cleaning it would have increased significantly.
embedding models from OpenAI may have given a higher accuracy. Have not tested with it.
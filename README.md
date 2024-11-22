# Lalo Model
This was created for my friend Lalo to identify the most important factors that predict interest rate cuts and to use these features to forecast rate cuts using a Random Forest Classification Model. The project leverages `fredapi`, `pandas`, `datetime`, and `scikit-learn` for data collection, feature engineering, and machine learning.

## How It Works:
- **Overview:**
  A Random Forest Classification model is a supervised machine learning algorithm that builds multiple decision trees during training and merges them to improve prediction accuracy and control overfitting. This approach is particularly well-suited for determining the importance of features in predicting complex outcomes, such as interest rate cuts, where relationships between indicators can be non-linear and interactive.

- **Data:**
  * Data for 49 different economic indicators is collected using FRED’s API going back to the 1970s. These indicators include metrics such as the unemployment rate, consumer sentiment, and housing starts, which provide a comprehensive set of macroeconomic data for the analysis.
  * If you are attempting to run the model yourself, please replace `"your FRED API Key here"` with your FRED API key in the `__main__` block of the code. A FRED API key can be acquired here: [https://fred.stlouisfed.org/docs/api/api_key.html].

- **Feature Engineering:**
  * The data for each indicator is cleaned and resampled using `pandas` to a consistent monthly frequency.
  * Missing values are filled using linear interpolation.
  * For each feature, four new features are generated:
      - Percentage Change (month-to-month).
      - Lagged values shifted by 1, 3, and 6 months.
  * The process results in a total of 245 features that are utilized in the model.

- **Recursive Feature Elimination:**  
  * Recursive Feature Elimination is employed to identify the most important features from the dataset. Using a Random Forest Model as the estimator, features are recursively removed, and the model’s performance is evaluated until the optimal set of features is determined.
  * This step is important as it reduces the model’s dimensionality, enhancing its computational efficiency. 

- **Random Forest Model:**  
  * A Random Forest Classification model is trained to predict whether an interest rate cut will occur next month.
  * Hyperparameter tuning (`GridSearchCV`) is used to find the optimal set of parameters for the model, including the number of trees, maximum tree depth, minimum number of samples needed for splitting nodes and leaf nodes, and class weightings to handle class imbalance between rate cuts and non-rate cuts.
  * The model is evaluated using precision, F1 score, and recall, focusing on recall to eliminate false negatives (missing a rate cut). 

---

## Output and Interpretation:
- **Best Hyperparameters:**
  * The best hyperparameters identified through `GridSearchCV` are:
    ```plaintext
    {'bootstrap': False, 'class_weight': {0: 1, 1: 4}, 'max_depth': 5, 
     'min_samples_leaf': 4, 'min_samples_split': 15, 'n_estimators': 100}
    ```

- **Classification Report:**
  * The model outputs a classification report, which includes statistics for both classes:
    - **Class 0:** No rate cut
    - **Class 1:** Predicted rate cut

  * The model outputs the following classification report:

    | Class           | Precision | Recall | F1-Score | Support |
    |-----------------|-----------|--------|----------|---------|
    | **0** (No Rate Cut) | 0.95      | 0.90   | 0.92     | 96      |
    | **1** (Rate Cut)    | 0.50      | 0.67   | 0.57     | 15      |
    | **Accuracy**        |           |        | 0.86     | 111     |
    | **Macro Avg**       | 0.72      | 0.78   | 0.75     | 111     |
    | **Weighted Avg**    | 0.88      | 0.86   | 0.87     | 111     |

- **Interpretation:**
  * **Precision:**
    - Class 0: 0.95 - 95% of predicted "no rate cut" instances are correct.
    - Class 1: 0.50 - 50% of predicted "rate cut" instances are correct.
  * **Recall:**
    - Class 0: 0.90 - 90% of actual "no rate cut" instances are correctly identified.
    - Class 1: 0.67 - 67% of actual "rate cut" instances are correctly identified.
  * **F1-Score:**
    - Class 0: 0.92 - A strong balance between precision and recall.
    - Class 1: 0.57 - Indicates room for improvement in predicting rate cuts.
  * **Support:**
    - Class 0: 96 instances of "no rate cut."
    - Class 1: 15 instances of "rate cut."
  * **Accuracy:**
    - The model correctly predicts outcomes **86%** of the time.
  * **Macro Average:**
    - An unweighted average of metrics for both classes:
      - Precision: 0.72
      - Recall: 0.78
      - F1-Score: 0.75
  * **Weighted Average:**
    - A weighted average considering class imbalance:
      - Precision: 0.88
      - Recall: 0.86
      - F1-Score: 0.87

- **Conclusion:**
  * The model is highly accurate in predicting "no rate cut" scenarios, with 95% precision and 90% recall. For predicting rate cuts, the model correctly identifies 67% of actual rate cuts, with 50% of its rate-cut predictions being accurate. Improvements in precision for rate cuts can enhance the model’s overall performance.

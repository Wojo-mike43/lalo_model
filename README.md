# Lalo Model
This was created for my friend Lalo to identify the most important factors that predict interest rate cuts and to use these features to forecast rate cuts using a Random Forest Classification Model. The projects leverages `fredapi`, `pandas`, `datetime`, and `scikit-learn` for data collection, feature engineering, and machine learning.

## How It Works:
- **Overview:**
A Random Forest Classification model is a supervised machine learning algorithm that builds multiple decision trees during training and merges them to improve prediction accuracy and control overfitting. This approach is particularly well-suited for determining the importance of features in predicting complex outcomes, such as interest rate cuts, where relationships between indicators can be non-linear and interactive.

- **Data:**
Data for 49 different economic indicators is collected using FRED’s API going back to the 1970s. These indicators include metrics such as the unemployment rate, consumer sentiment, and housing starts, which provide a comprehensive set of macroeconomic data for the analysis.

- **Feature Engineering:**
  * The data for each indicator is cleaned and resampled using `pandas` to a consistent monthly frequency.
  * Missing values are filled using linear interpolation.
  * For each feature, four new features are generated:
      - Percentage Change (month-to-month).
      - Lagged values are shifted by 1, 3, and 6 months.
  * The process results in a total of 245 features that are utilized in the model.

- **Recursive Feature Elimination**  
  * Recursive Feature Elimination is employed to identify the most important features from the dataset. Using a Random Forest Model as the estimator, features are recursively removed, and the model’s performance is evaluated until the optimal set of features is determined.
  * This step is important as it reduces the dimensionality of the model, and enhances its computational efficiency. 

- **Random Forest Model:**  
  * Next, a Random Forest Classification model is trained to predict whether an interest rate cut will occur in the next month.
  * Hyperparameter tuning (GridSearchCV) is used to find the optimal set of parameters for the model, including the number of trees, maximum tree depth, minimum number of samples needed for splitting nodes and leaf nodes, and class weightings to handle class imbalance between rate cuts and non-rate cuts.
  * The model is evaluated using precision, F1 score, and recall, with a focus on recall to eliminate false negatives (missing a rate cut). 

- **Output and Interpretation**
  * 

  
  

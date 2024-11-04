# Credit-Card-Fraud-Detection

# **Overview**
This project is a credit card fraud detection system using machine learning techniques to predict fraudulent transactions. By leveraging Logistic Regression and a structured ETL pipeline, this project aims to assist financial institutions in identifying potentially fraudulent transactions, minimizing losses, and protecting customers.

# **Table of Contents**
- Project Description
- Dataset
- Data Preprocessing
- Machine Learning Model
- Evaluation Metrics
- Results and Insights
- Conclusion
- How to Use
- Acknowledgments

# **Project Description**
In the complex field of credit card fraud, identifying fraud patterns is crucial to preventing financial crimes. This project explores the use of machine learning, specifically Logistic Regression, for detecting fraudulent transactions. The goal is to build a model that generalizes well, reducing the risk of overfitting, and can effectively flag potentially fraudulent transactions in real-world applications.

# **Dataset**
The dataset used for this project is the Credit Card Fraud Detection dataset, available on Kaggle. It includes anonymized transaction data with 492 fraudulent transactions out of 284,807 total transactions. Key characteristics:

- **Class Imbalance:** Fraudulent transactions make up only 0.172% of the total.
- **Features:** The dataset includes 28 anonymized transaction features, a "Time" column,id column,transaction amount and class.
# **Data Preprocessing**
A series of data cleaning and preprocessing steps were applied to ensure the model performs accurately:

- **Handling Missing Values:** Verified that no null values were present in the dataset.
- **Data Normalization:** Standardized the "Amount" feature to align it with other scaled features.
- **Balancing the Dataset:** Used undersampling to address the class imbalance.
- **Data Transformation:** Used SparkSQL for aggregation, grouping, and time-based analysis of transaction patterns.

# **Machine Learning Model**
The model employs Logistic Regression due to its interpretability and efficiency for binary classification. To improve the model's performance, Random Search Cross-Validation was used for hyperparameter tuning. Key aspects of the modeling process:

- **Pipeline Integration:** A pipeline was used to ensure seamless data preprocessing and training steps.
- **Hyperparameter Tuning:** Optimized the model by tuning parameters like the regularization term.
- **Subset Sampling:** Used a subset of the data during tuning for computational efficiency.

# **Evaluation Metrics**
The model's effectiveness was evaluated using several metrics:

- **Accuracy:** Measures overall correctness.
- **Precision and Recall:** Assesses the model’s ability to identify fraud correctly.
- **AUC (Area Under the ROC Curve):** Indicates how well the model distinguishes between fraud and non-fraud.
- **R² Score:** Reflects the variance explained by the model.

# **Results and Insights**
- **High Initial Accuracy:** Initial accuracy was 95%, though some overfitting was observed.
- **Performance after Tuning:** The model became more generalized with slightly lower performance metrics after tuning, providing a more realistic assessment of its ability to generalize.

# **Insights from Data Analysis:**
- Fraudulent transactions often occur in clusters with similar amounts.
- High variability features and outliers are key indicators of fraud.
 # Conclusion
This project demonstrates an effective approach for credit card fraud detection using Logistic Regression. The tuned model balances sensitivity and generalization, making it robust for real-world scenarios. This solution showcases how machine learning can support financial institutions in improving fraud detection, enhancing security, and reducing financial risks.

# How to Use
**Clone the Repository:**

git clone https://github.com/tamannada26/CreditCardFraudDetection.git

- **Run the Code:** Follow the provided Jupyter notebooks or scripts to preprocess the data, train the model, and evaluate its performance.
- **Dependencies:** Ensure PySpark, Pandas, Scikit-learn, and other dependencies are installed.
# Acknowledgments
- **Dataset Source:** Kaggle - Credit Card Fraud Detection Dataset
- **Inspiration:** This project was inspired by the challenge of building scalable, accurate fraud detection systems to combat financial crime.

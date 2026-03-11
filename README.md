# Credit-Card-Fraud-Detection-Dataset-2013

### Overview
This project focused on building a machine learning system capable of detecting fraudulent credit card transactions. Fraud detection is a critical application of machine learning because fraudulent activities are rare but financially damaging. This creates a highly imbalanced classification problem where traditional evaluation metrics such as accuracy can be misleading.

The objective of this project was to develop and evaluate several classification models capable of identifying fraudulent transactions while minimizing false alarms.

The goal is to develop a model that can accurately identify fraudulent transactions while minimizing false alarms.

### Dataset

The dataset used in this project contains anonymized credit card transactions. Due to privacy concerns, most features have been transformed using Principal Component Analysis.

Key characteristics of the dataset:
1. Transactions are represented by features V1 to V28
These features are principal components derived from the original confidential variables
2. The dataset is highly imbalanced, with fraudulent transactions representing a very small fraction of the total data
3. Target variable:
Value	Meaning
0	Legitimate transaction
1	Fraudulent transaction

### Project Workflow
The project follows a structured machine learning pipeline:
1. Problem Definition
2. Data Loading and Inspection
3. Understanding the target variable and EDA
4. Train, Test Split + Scaling
5. Building a baseline models (without handling imbalance) and evaluate
6. Handle imbalance
7. Train models again and evaluate
8. Hyperparameter tuning with cross-validation
9. Threshold tuning
10. Feature Importance
11. Confusion Matrix
11. Summary, Model interpretation and Conclusion.

### Models Implemented
Three classification algorithms were evaluated:
1. Logistic Regression
2. Random Forest
3. XGBoost

### Handling Class Imbalance
Fraudulent transactions account for a very small proportion of the dataset. This imbalance can cause models to become biased toward predicting the majority class.

To address this issue, the SMOTE technique was applied to the training data. SMOTE generates synthetic samples of the minority class, helping the model learn patterns associated with fraudulent transactions more effectively.

### Model Evaluation Metrics
Because the dataset is highly imbalanced, accuracy alone is not sufficient to evaluate model performance. The following metrics were used:
1. Precision – how many predicted fraud cases were actually fraud
2. Recall – how many fraud cases were successfully detected
3. F1 Score – harmonic mean of precision and recall
4. ROC-AUC – overall ability of the model to distinguish between classes

### Baseline Model Results
| Model	Accuracy | Precision | Recall | F1 Score | ROC-AUC |	
|-------|---------|--------------|----------------|------------------|
| Logistic Regression	| 0.999157 |	0.828947 |	0.642857 |	0.724138 |	0.955898 |
| Random Forest	| 0.999596 |	0.941176 |	0.816327 |	0.874317 |	0.975394 |
| XGBoost |	0.999438	| 0.866667 |	0.795918 |	0.829787 |	0.938952 |

Among the baseline models, Random Forest provided the strongest balance between precision and recall.

### Model Results After Handling Imbalance
| Model	Accuracy | Precision | Recall | F1 Score | ROC-AUC |	
|-------|---------|---------------------|------------------|------------------|
| Logistic Regression	| 0.974211 |	0.058027 |	0.918367 |	0.109157 |	0.969863 |
| Random Forest	| 0.999491 |	0.863158 |	0.836735 |	0.849741 |	0.975394 |
| XGBoost |	0.998894	| 0.631579 |	0.857143 |	0.829787 |	0.980526 |

Applying SMOTE improved the model’s ability to detect fraudulent transactions, particularly by increasing recall.

### Threshold Optimization
Classification models typically use a default decision threshold of 0.5. However, this threshold may not provide the best balance between fraud detection and false alarms.
To address this, threshold tuning was performed using the Precision-Recall Curve. This allowed the model to achieve a more suitable balance between precision and recall for fraud detection.

Final results for the optimized model:
| Class | Precision | Recall | F1 Score |
|-------|---------|---------------------|------------------|
| Normal Transactions	| 1.00	| 1.00	| 1.00 |
| Fraudulent Transactions	| 0.96	| 0.78	| 0.86 |

Overall ROC-AUC: 0.975

### Key Insights
Several important insights emerged from this analysis:
1. Accuracy alone is not a reliable metric for highly imbalanced datasets.
2. Addressing class imbalance significantly improves fraud detection capability.
3. Tree-based ensemble models performed better than linear models in this problem.
4. Threshold tuning helped achieve a better balance between detecting fraud and minimizing false positives.

### Tech Stack
1. Python
2. Pandas
3. NumPy
4. Matplotlib
5. Scikit-learn
6. XGBoost
7. Imbalanced-learn

### Future Improvements
Possible extensions for this project include:
1. Hyperparameter tuning for model optimization
2. Implementing additional ensemble models
3. Deploying the model as a fraud detection API
4. Testing the model on other financial fraud datasets

### How to Run
1. Clone the repository
2. Install required packages: numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn
3. Run the notebook step by step to replicate results.

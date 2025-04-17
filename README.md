Stroke Prediction Model 🩺

📖 Overview
This project develops a machine learning model to predict stroke likelihood using patient data. Built with a robust pipeline, it handles numerical and categorical features, addresses missing values, and mitigates class imbalance. The model employs Linear Discriminant Analysis (LDA) for classification, delivering reliable predictions.

📊 Dataset
The model is trained on stroke-data.csv, containing patient attributes relevant to stroke prediction. Key features include:
Features

Categorical:
hypertension: Presence of hypertension (0 or 1)
heart_disease: Presence of heart disease (0 or 1)
ever_married: Marital status (Yes/No)
work_type: Occupation type (e.g., Private, Self-employed)
Residence_type: Urban or Rural
smoking_status: Smoking habits (e.g., never smoked, smokes)


Numerical:
avg_glucose_level: Average blood glucose level
bmi: Body Mass Index
age: Patient age


Target:
stroke: Stroke occurrence (0 or 1)


Dropped:
id: Unique patient identifier (not used for prediction)



The load_data() function loads the dataset, separates features (X) and target (y), and identifies feature types for preprocessing.

🛠️ Model Pipeline
The model is implemented using a scikit-learn Pipeline for streamlined preprocessing and classification:

ColumnTransformer:
Numerical: Imputes missing values with median (SimpleImputer(strategy='median')).
Categorical: Applies one-hot encoding (OneHotEncoder(handle_unknown='ignore')) to handle unseen categories.


PowerTransformer:
Uses Yeo-Johnson transformation to normalize numerical features (standardize=True).


SMOTE:
Addresses class imbalance by generating synthetic samples for the minority class (stroke cases).


Linear Discriminant Analysis (LDA):
Classifies patients based on feature combinations optimized for class separation.




📈 Model Evaluation
The model is evaluated using Repeated Stratified K-Fold Cross-Validation:

Setup: 10 folds, 3 repeats (RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)).
Metric: ROC AUC, ideal for imbalanced datasets.
Visualization: A boxplot displays the distribution of ROC AUC scores, with the mean score highlighted.

The evaluate_model() function computes cross-validation scores, providing insights into model performance.

💾 Model Training & Saving
The pipeline is trained on the entire dataset and saved as stroke_prediction_model.joblib for future use, such as predictions or deployment.

🧰 Dependencies
The following libraries are required:

pandas: Data manipulation
numpy: Numerical operations
scikit-learn: Machine learning components
imblearn: Class imbalance handling
joblib: Model persistence
matplotlib: Visualization

Install them with:
pip install pandas numpy scikit-learn imblearn joblib matplotlib


🚀 Usage
Training the Model

Place stroke-data.csv in the project directory.
Run the script to:
Load and preprocess data
Evaluate the model (view ROC AUC boxplot)
Train and save the final model (stroke_prediction_model.joblib)



Making Predictions
Use the saved model for new predictions:
from joblib import load
import pandas as pd

# Load the model
model = load('stroke_prediction_model.joblib')

# Example new data
new_data = pd.DataFrame({
    'age': [45],
    'hypertension': [0],
    'heart_disease': [0],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [85.5],
    'bmi': [25.0],
    'smoking_status': ['never smoked']
})

# Predict
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]
print(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")
print(f"Stroke Probability: {probability[0]:.3f}")


📉 Results
The model's ROC AUC scores from cross-validation are visualized in a boxplot, showing performance consistency. The mean ROC AUC reflects the model's ability to distinguish stroke cases, with the standard deviation indicating variability.

🔮 Future Improvements

Feature Engineering: Add derived features or interactions.
Hyperparameter Tuning: Optimize LDA or test other classifiers (e.g., XGBoost).
Oversampling Alternatives: Explore ADASYN or combined sampling methods.
Data Quality: Address outliers or inconsistent values.
Deployment: Create a web API with Flask or FastAPI.


📜 License
This project is licensed under the MIT License.

📬 Contact
For questions or contributions, reach out to [Your Name] at [Your Email].

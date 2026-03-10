# QSAR Modeling of MAO-B Inhibitors

## Goal
Development of a QSAR model for predicting the activity and classification of MAO-B inhibitors to support the search for potential drugs for neurodegenerative and psychiatric disorders.

## Dataset
Source: https://www.ebi.ac.uk/chembl/explore/activities/STATE_ID:FE6GtvQdf2VN4tQ5eJ05Ew%3D%3D

# Stages
EDA + Preprocessing: data analysis, gap handling, visualizations
Baseline Model: linear regression, first estimation
Final Model + Explainability: metrics + SHAP
Streamlit app

## Results
Baseline: R2: 0.547, Final model: R2: 0.601

## Conclusion
The model was improved through hyperparameter tuning.

# Streamlit app "BioActivity Predictor (pChEMBL Prediction)"
A Streamlit-based app for calculating the biological activity of molecules. It uses a combination of RDKit descriptors, Morgan SVD-compressed fingerprints, and the CatBoost model. app.py — web interface code.
final_model.pkl — trained model file.
svd_transformer.pkl — fingerprint processing transformer.
feature_names.pkl — list of features for the model.

## Installation and Run
Run the following command in the terminal to install all dependencies:
pip install streamlit pandas numpy rdkit catboost joblib scikit-learn
Run the application:
streamlit run app.py

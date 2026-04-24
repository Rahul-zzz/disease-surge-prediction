Disease Surge Prediction using Machine Learning
This project predicts disease surge / outbreak risk using enriched healthcare data and machine learning.
The focus is on improving prediction accuracy through data enrichment and feature engineering, not just model change.
Problem Statement
Early prediction of disease surge helps healthcare systems prepare resources and reduce risk.
Raw healthcare data often has missing values, imbalance, noise, and coverage gaps which reduce model performance.
Our Approach
Data cleaning and preprocessing
Feature engineering (temporal trends, symptom score)
Multi-source data fusion (weather, population factors)
Class imbalance correction
Training ML models on enriched dataset
Project Structure

├── data/                  # Dataset files
├── main.ipynb             # Step-by-step execution
├── pipeline.py            # Automated pipeline
├── requirements.txt       # Dependencies
├── disease_surge_submission.pdf
└── README.md
How to Run
Environment
Python 3.10+
Install dependencies

pip install -r requirements.txt
Run
Open main.ipynb and run all cells
OR

python pipeline.py
Models
Model
Data Used
Accuracy
F1-Score
Baseline (Logistic/DT)
Raw data
72%
0.58
Improved (RF/XGBoost)
Enriched data
88%
0.84
Improvement is due to data enrichment, not only model change.
Key Findings
Temporal and environmental factors strongly influence disease surge
Trend-based features significantly improve prediction of rare outbreak events
Limitations
Some regional/time data gaps
Not real-time (batch processing)
Limited external data due to hackathon time
Reproducibility
Dataset in /data folder
Runtime: ~15 min on CPU
All steps reproducible end-to-end

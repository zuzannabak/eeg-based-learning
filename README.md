# EEG-Based Learning Outcome Prediction

This project investigates whether EEG band-power features can predict a student's comprehension of instructional material.  
The work was completed as part of a Machine Learning course at Temple University.

## Tech Stack
**Python 3.10+**, NumPy, Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), Matplotlib / Seaborn, Jupyter Notebook


## Project Overview
The goal was to classify **understood vs. not understood** based on short EEG segments.  
Using a **subject-wise split**, I compared two model families:

- **Feed-Forward Neural Network (FNN / MLPClassifier)**
- **XGBoost**

I also experimented with class imbalance strategies:
- baseline  
- class weighting  
- SMOTE  


## Key Findings
- **FNN baseline** was the only model that learned a meaningful signal  
  → ROC-AUC = **0.704**
- **XGBoost baseline** achieved high accuracy  
  → but AUC < 0.50 (predicts almost all samples as class 1)
- Class weighting and SMOTE did **not** improve separability.

Full results and confusion matrices are included in the final report.


## Project Structure
```yaml
EEG-BASED-LEARNING/
│
├── data/
│ ├── raw/ # original EEG dataset (from Kaggle)
│ └── processed/ # cleaned and preprocessed data
│
├── 1_EDA.ipynb # exploratory data analysis
├── 2_Modeling.ipynb # FNN + XGBoost modeling pipeline
├── 3_ClassImbalance_FNN_XGBoost.ipynb # imbalance strategies
│
├── final_project_report.pdf # 4-page final ML report
└── requirements.txt
```



## Dataset
📥 The dataset comes from Kaggle:  
**EEG Brainwave Dataset — Predict Student Understanding**  
https://www.kaggle.com/datasets/madyanomar/eeg-data-distance-learning-environment

(Accessed: Nov 7, 2025)



## Final Report
The full analysis, results, discussion, and model comparison are summarized in:  
**`final_project_report.pdf`**



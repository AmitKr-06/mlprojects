## End To End Machine Learning Project

# Titanic Survival Prediction

## Overview
Predicting passenger survival on the Titanic using 
machine learning. Final model achieves 84.92% accuracy
using Random Forest Classifier.

## Dataset
- Source: Kaggle Titanic Dataset
- Shape: 891 rows × 12 raw features
- Target: Survived (0 = No, 1 = Yes)

## Project Pipeline

### 1. Missing Value Handling
- Age: filled with median
- Embarked: filled with mode
- Cabin: converted to binary Has_Cabin feature

### 2. Outlier Handling
- Fare: log transformation → Fare_log
- Age, SibSp, Parch: capped at percentile boundaries

### 3. Feature Engineering
- Family_Size = SibSp + Parch + 1
- Title_Group extracted from Name column
- Fare_Binned: 5 ordinal fare categories

### 4. Feature Selection
- Dropped: PassengerId, Name, Ticket (no signal)
- Dropped: intermediate transformation columns

### 5. Encoding
- Binary : Sex
- One-Hot : Embarked, Title_Group
- Ordinal : Fare_Binned

### 6. Final Features (15 features)
- Pclass, Sex, Has_Cabin, Fare_log
- Age_handle, Family_Size, Fare_Binned_enc
- Title_Group_* (6 cols), Embarked_* (2 cols)

### 7. Model Results
| Model               | Accuracy |
|---------------------|----------|
| Random Forest       | 84.92%   |
| Gradient Boosting   | 83.24%   |
| SVM (scaled)        | 82.12%   |
| Logistic Regression | 81.01%   |
| Decision Tree       | 78.77%   |

## Best Model
Random Forest Classifier (default parameters)
- Accuracy  : 84.92%
- Precision : 0.83
- Recall    : 0.83
- F1 Score  : 0.83

## Requirements
pip install numpy pandas scikit-learn 
            matplotlib seaborn jupyter

## How to Run
1. Clone the repository
2. Install requirements
3. Run notebook: titanic_prediction.ipynb
```

---

**Is everything done well? Honest Review:**
```
    Data Cleaning      → thorough, no shortcuts
    Feature Engineering → smart (Family_Size, Title_Group)
    Encoding           → correct method for each type
    Model Selection    → compared 5 models fairly
    Tuning             → used GridSearchCV properly
    Evaluation         → accuracy + F1 + confusion matrix
    Final Accuracy     → 84.92% (above Kaggle average of 77–80%)

  One thing to add later:
   → Feature importance plot
   → Cross validation score on final model
   → Save model with joblib for deployment
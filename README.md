# CancerPredictionMLModel

This repository contains a complete machine learning pipeline built from the ground up to predict whether a patient is likely to have cancer based on several lifestyle, biological, and historical health features.

## 📊 Dataset

The dataset consists of 1,500 patient records with the following features:
- `Age`: Patient's age
- `Gender`: Binary encoded
- `BMI`: Body Mass Index
- `Smoking`: Binary indicator
- `GeneticRisk`: Genetic predisposition score
- `PhysicalActivity`: Hours of physical activity per week
- `AlcoholIntake`: Units of alcohol consumed
- `CancerHistory`: Family history of cancer
- `Diagnosis`: Target variable (1 = has cancer, 0 = does not)

## ⚙️ Workflow

1. **Data Preprocessing**
   - Loaded and cleaned the CSV file.
   - Applied iterative cleaning through a `dataPreperation()` function.
   - Normalized data using Min-Max Scaling.

2. **Balancing Classes**
   - Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target classes for better model performance.

3. **Model Training & Evaluation**
   - Models used:
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Random Forest
     - XGBoost
   - Performance metrics:
     - Accuracy on both training and testing datasets
     - Confusion Matrix
     - Probability Predictions

4. **Results**
   - Best performing models:
     - **Random Forest** and **XGBoost** achieved over **91% test accuracy**.
   - Visualization:
     - Confusion matrix heatmaps
     - Output probabilities and final predictions

## 📦 Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib
- imbalanced-learn

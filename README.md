# AI-Based Multi-Disease Prediction System

## Overview
This project is a machine learning-based web application for predicting:
- Heart Disease
- Diabetes
- Breast Cancer
- Liver Cirrhosis Stage

The system is developed using Python, Scikit-learn, XGBoost, and Django.

---

## Datasets Used

### 1. Heart Disease
Source: UCI Heart Disease Dataset  
Instances: 920 records  

### 2. Diabetes
Source: PIMA Indians Diabetes Dataset (Kaggle)  
Instances: 768 records  

### 3. Breast Cancer
Source: Wisconsin Breast Cancer Dataset (UCI)  
Instances: 569 records  

### 4. Liver Cirrhosis
Source: Kaggle – Liver Cirrhosis Stage Detection Dataset  
License: MIT License  
Instances: 25,000 records  
Original Source: Mayo Clinic Primary Biliary Cirrhosis Study  

---

## Algorithms Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- XGBoost

Best models selected based on weighted F1-score using 10-fold Stratified Cross Validation.

---

## Deployment
The system is deployed as a Django web application using PostgreSQL database.

---

## Author
Final Year B.Tech Project – Electronics and Communication Engineering
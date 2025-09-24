# 💼 Salary Prediction Model with Streamlit Deployment

This project demonstrates a complete machine learning workflow to predict salaries based on years of experience using **Simple Linear Regression**. It includes data preprocessing, model training, performance evaluation, visualization, and deployment via an interactive Streamlit app.

---

## 📌 Project Objectives

- Build a regression model to predict salary from experience
- Visualize model performance on training and test sets
- Evaluate accuracy using R² and Mean Squared Error
- Save the trained model for reuse with `pickle`
- Deploy a user-friendly app using Streamlit for real-time predictions

---

## 📊 Dataset

- **Source**: [Salary_Data.csv]
- **Features**:
  - `YearsExperience`: Independent variable
  - `Salary`: Dependent variable

---

## 🚀 Workflow Overview

1. **Data Loading**: Read CSV using `pandas`
2. **Preprocessing**: Split into training and test sets (80/20)
3. **Model Training**: Fit a `LinearRegression` model
4. **Evaluation**:
   - R² Score for bias/variance
   - Mean Squared Error for residuals
5. **Visualization**: Plot training and test predictions using `matplotlib`
6. **Prediction**: Estimate salary for custom inputs (e.g., 12, 20 years)
7. **Model Saving**: Serialize with `pickle`
8. **App Deployment**: Build a Streamlit app for interactive use

---

## 🖥️ Streamlit App

The app allows users to input years of experience and receive a predicted salary instantly. It loads the pickled model and displays results with clean formatting.

### 🔧 To Run Locally:
```bash
streamlit run app/streamlit_app.py

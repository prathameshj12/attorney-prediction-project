# Attorney Involvement Prediction in Insurance Claims

A machine learning + Streamlit powered tool to predict whether an insurance claim is likely to involve an attorney. This assists insurers in proactively flagging claims that may require legal attention based on the claimant, accident, and policy details.

---

## Streamlit App Snapshot

![WhatsApp Image 2025-03-29 at 17 56 29_48f1d97c](https://github.com/user-attachments/assets/9ae6174a-1c96-47f6-bd29-4a0642a42a00)

---

## Project Overview

- **Domain**: Insurance / Legal Analytics  
- **Problem Type**: Classification  
- **Tech Stack**: Python, Pandas, Scikit-learn, XGBoost, Streamlit 

## Features

- Predicts attorney involvement based on user inputs.
- Interactive Streamlit interface with clear visualization of prediction and confidence.
- Uses feature-engineered pipeline and a trained ML model.

---

## Workflow Summary

1. **Data Preprocessing**: Handled missing values, encoded categorical variables, and scaled numerical features.
2. **Exploratory Data Analysis (EDA)**: Uncovered insights on claim patterns and attorney involvement.
3. **Model Training**: Compared models like Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), XGBoost, LightGBM and Neural Network.
4. **Evaluation**: Used accuracy, precision, recall, and ROC-AUC to evaluate performance.
5. **Prediction**: Built a final model pipeline to make predictions on unseen data.
6. **Deployment**: Integrated final model into a Streamlit app.

---

## Key Highlights

- Robust model pipeline with PCA & hyperparameter tuning.
- Clean Streamlit UI for real-time predictions.
- No confidential data included. Dataset not provided due to sensitivity.

---

## Try It Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/attorney-involvement-prediction.git
cd attorney-involvement-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

> You can also test the model directly via the notebook.

---

## Folder Structure

```
.
├── app.py                             # Streamlit app
├── Attorney_Involvement_in_Insurance_Claims_GitHub.ipynb
├── assets/
│   └── attorney_predictor_ui.jpg      # Screenshot of the UI
├── README.md
├── requirements.txt
```

---

## Acknowledgements

This was created as part of a real-world insurance analytics project. All sensitive content has been sanitized while preserving technical integrity.

---

*Feel free to fork, star, and reach out for improvements or collaborations!*

# Attorney Involvement Prediction in Insurance Claims

A machine learning + Streamlit powered tool to predict whether an insurance claim is likely to involve an attorney. This assists insurers in proactively flagging claims that may require legal attention based on the claimant, accident, and policy details.

---

## Streamlit App Snapshot

![Streamlit Screenshot](assets/attorney_predictor_ui.jpg)

---

## Project Overview

- **Domain**: Insurance / Legal Analytics  
- **Problem Type**: Classification  
- **Tech Stack**: Python, Pandas, Scikit-learn, XGBoost, Streamlit 

## Features

- Predicts attorney involvement based on user inputs.
- Interactive Streamlit interface with clear visualization of prediction and confidence.
- Uses feature-engineered pipeline and a trained ML model (`best_model.pkl`).

---

## Workflow Summary

1. **Preprocessing**: Imputation, encoding, feature scaling.
2. **EDA**: Explored patterns of attorney involvement.
3. **Modeling**: Trained multiple models (Logistic Regression, XGBoost, Neural Network, etc.)
4. **Evaluation**: Accuracy, Precision, Recall, ROC-AUC.
5. **Deployment**: Integrated final model into a Streamlit app.

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
├── best_model.pkl                     # Final trained model pipeline
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

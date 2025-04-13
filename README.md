# Attorney Involvement Prediction in Insurance Claims

This project builds a machine learning model to predict whether an insurance claim is likely to involve an attorney. The goal is to help insurers proactively identify such claims based on various features like claimant details, accident information, and claim attributes.

## Project Overview

- **Domain**: Insurance / Legal Analytics  
- **Problem Statement**: Predict if an attorney will be involved in an insurance claim.  
- **Type**: Classification  
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Workflow Summary

1. **Data Preprocessing**: Handled missing values, encoded categorical variables, and scaled numerical features.
2. **Exploratory Data Analysis (EDA)**: Uncovered insights on claim patterns and attorney involvement.
3. **Model Training**: Compared models like Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), XGBoost, LightGBM and Neural Network.
4. **Evaluation**: Used accuracy, precision, recall, and ROC-AUC to evaluate performance.
5. **Prediction**: Built a final model pipeline to make predictions on unseen data.

> *Note: This project was adapted from an internship. Some dataset descriptions and results have been anonymized or modified to comply with confidentiality.*

## Folder Structure

```
.
├── Attorney_Involvement_in_Insurance_Claims_GitHub.ipynb # Final notebook
├── data/
│   └── Dataset.csv                             # (Optional) Dummy dataset if added
├── README.md
```

## Key Highlights

- Achieved robust performance on unseen test data.
- Employed pipeline and hyperparameter tuning for production-ready predictions.
- Visualizations to explain model behavior and class distribution.

## How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/attorney-involvement-prediction.git
   cd attorney-involvement-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   Open `Attorney_Involvement_in_Insurance_Claims_GitHub.ipynb` in Jupyter or VSCode and follow along.


## Acknowledgements

This project was developed as part of a real-world insurance analytics initiative. Key methodologies and insights have been retained, while sensitive details have been anonymized to respect confidentiality.

---

**Feel free to fork, improve, or reach out with feedback. Happy coding!**

# CustomerIQ — Customer Churn Prediction

An end-to-end Data Science project that analyses retail customer behaviour, predicts churn risk, and explains predictions through an interactive dashboard.

## Features
- Data cleaning and customer behaviour analysis
- Customer churn prediction using Logistic Regression and Random Forest
- Baseline comparison and temporal evaluation
- SHAP-based model explanations
- Interactive dashboard with customer search, risk filters, charts, and CSV downloads
- Churn sensitivity and customer-level evaluation experiments

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · SHAP · Streamlit · Plotly

## Model Results
- Selected model: Logistic Regression
- Temporal test ROC-AUC: **0.7435**
- Temporal test accuracy: **68.1%**
- Test cohort: 3,616 customer snapshots

Additional experiments compare simple baselines, test customer-level generalisation, and evaluate 30-, 60-, 90-, and 120-day churn definitions.

## Dataset
[UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

Download the dataset and place `Online Retail.xlsx` in `data/raw/`.

## Run Locally

```bash
python -m pip install -r requirements.txt
python src/clean_data.py
python src/build_training_dataset.py
python src/train_advanced_model.py
python src/explain_model.py
python -m streamlit run app/dashboard.py
```

Run the scripts in order. Run `python src/inspect_data.py` before cleaning if you want to inspect the dataset first.

## Limitations
Churn is defined as no qualifying purchase within the next 60 days. Risk scores are not calibrated probabilities, and results from historical data may not generalise to current customers.

## Author
**Vaishnavi Aher**  
B.Tech Data Science · MIT Academy of Engineering

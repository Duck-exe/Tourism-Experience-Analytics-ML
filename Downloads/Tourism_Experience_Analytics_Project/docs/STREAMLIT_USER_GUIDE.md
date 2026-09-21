# Streamlit User Guide

## Start the application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Executive Dashboard
Review transaction count, users, historical attractions, average rating, visit-mode distribution, rating distribution, yearly trend, top attractions and attraction-type behavior.

## Visit Mode Prediction
Choose user geography, visit year/month and a historical attraction. Click **Predict Visit Mode** to receive the predicted class and class probabilities.

## Rating Prediction
Choose the same profile plus a visit mode. Click **Predict Rating** to receive a clipped 1-5 predicted satisfaction score.

## Recommendations

### Known User
Choose a historical `UserId`, recommendation strategy and list size.
- `content`: strongest MAP@10 in the supplied evaluation and supports the full expanded catalog.
- `hybrid`: blends content, collaborative and popularity signals.
- `collaborative`: emphasizes rating-history similarity; evidence exists only for 30 historically rated attractions.

### New / Cold-Start User
Choose preferred attraction type, destination country/city and optional visit mode. The app ranks attractions without requiring a historical user ID.

## Model Evaluation
Shows the comparison tables and held-out metrics for classification/regression plus recommendation RMSE and MAP@10.

## SQL Analytics
Choose a prewritten stakeholder query and inspect the SQL and returned table. The full query collection is also in `sql/analysis_queries.sql`.

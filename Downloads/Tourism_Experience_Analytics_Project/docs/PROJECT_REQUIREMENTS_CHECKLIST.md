# Project Requirements Checklist

This checklist maps the supplied project brief to concrete files in this submission.

| Requirement | Status | Implementation / evidence |
|---|---|---|
| Data cleaning: missing values | Complete | `src/data_pipeline.py`, `data/processed/cleaning_report.json` |
| Resolve categorical discrepancies | Complete | Placeholder geography normalized; mixed expanded attraction-type field preserved + standardized labels |
| Date/time consistency | Complete | `VisitYear` and `VisitMonth` validated and typed; month constrained 1-12 |
| Handle rating errors/outliers | Complete | Rating constrained to 1-5; invalid rows would be excluded |
| Encode categorical variables | Complete | One-hot preprocessing for linear models; ordinal preprocessing for tree/XGBoost models |
| Aggregate user profile information | Complete | Recommendation user profiles from historical item/rating interactions |
| Join transaction, user, city, attraction data | Complete | `tourism_merged_clean.csv`, 100% user/attraction historical join coverage |
| Normalize/scale numerical model inputs | Complete | Linear-model pipelines standardize numeric year/month; tree models do not require scaling |
| EDA: user geography | Complete | Dashboard + figures + SQL by continent/region/country |
| EDA: attraction type/popularity | Complete | Static figures, SQL outputs, Streamlit dashboard |
| EDA: VisitMode vs user behavior | Complete | Visit-mode distributions/ratings + prediction model |
| EDA: rating distribution | Complete | `01_rating_distribution.png` + dashboard |
| Visualization: trends | Complete | yearly/monthly charts + SQL |
| Visualization: patterns/outliers | Complete | rating, visit mode, attraction/type, geography and year/month views |
| SQL | Complete | `database/tourism.db`, `sql/analysis_queries.sql`, app SQL Analytics page |
| Regression model | Complete | Ridge, Random Forest, XGBoost comparison; XGBoost selected |
| Regression metrics R²/MSE/RMSE/MAE | Complete | `regression_model_comparison.csv`, `regression_test_metrics.json` |
| Classification model | Complete | Logistic Regression, Random Forest, XGBoost comparison; XGBoost selected |
| Classification accuracy/precision/recall/F1 | Complete | `classification_test_metrics.json`, `classification_report.json` |
| Model comparison | Complete | classification + regression comparison CSV files and app page |
| Collaborative filtering | Complete | `src/recommender.py` |
| Content-based filtering | Complete | `src/recommender.py`; supports full expanded catalog |
| Hybrid system (optional) | Complete | Blend of content/collaborative/popularity |
| Recommendation RMSE | Complete | `recommendation_metrics.json` |
| Recommendation MAP | Complete | MAP@10 for collaborative/content/hybrid |
| Streamlit visit-mode prediction | Complete | `app.py` |
| Streamlit recommendations | Complete | known-user + cold-start pages in `app.py` |
| Popular attractions/top regions/user segments | Complete | Executive Dashboard and SQL Analytics |
| Cleaned dataset deliverable | Complete | `data/processed/tourism_merged_clean.csv`, `attraction_catalog_clean.csv` |
| Source-code deliverable | Complete | `src/`, `app.py`, `run_pipeline.py` |
| Application deliverable | Complete | Streamlit application + Dockerfile |
| Documentation/report | Complete | `FINAL_PROJECT_REPORT.md`, `FINAL_PROJECT_REPORT.docx`, README, data dictionary, user guide |
| Actionable business insights | Complete | Report sections on segmentation, attraction prioritization, cold-start strategy and data gaps |
| Code quality | Complete | modular reusable code, comments/docstrings, centralized paths, automated tests |
| Presentation clarity | Complete | report + `PRESENTATION_NOTES.md`; key findings and recommendations are separated from technical details |

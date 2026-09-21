# Tourism Experience Analytics

**Classification, Prediction, and Recommendation System**

This repository is a complete implementation of the supplied project brief. It covers data cleaning and preprocessing, SQL analytics, EDA and visualization, rating regression, visit-mode classification, collaborative/content/hybrid recommendations, model comparison and evaluation, and a Streamlit application.

## 1. Project objectives covered

1. **Regression:** predict a tourist's attraction rating (1-5).
2. **Classification:** predict visit mode: Business, Couples, Family, Friends, or Solo.
3. **Recommendation:** rank personalized attractions using collaborative filtering, content-based filtering, and an optional hybrid strategy.
4. **Tourism analytics:** expose trends by user geography, visit mode, attraction type, time, and popularity.
5. **Deployment:** provide a user-friendly Streamlit app with predictions, recommendations, dashboards, model metrics, and SQL analytics.

## 2. Dataset and cleaning summary

- 52,930 transaction rows retained after validation.
- 33,530 users.
- 30 historically rated attractions (all transaction attraction IDs resolve correctly).
- 1,698 sites in the expanded `Updated_Item.xlsx` catalog.
- 4 missing user `CityId` values are mapped to the dataset's placeholder ID `0` and presented as `Unknown`.
- Rating values are validated to the 1-5 range and visit months to 1-12.
- The expanded attraction file contains **1,668 text attraction types mixed into the `AttractionTypeId` field** (e.g., Beach, Museum, Park, Temple). The pipeline preserves the raw field and creates standardized textual `AttractionTypeLabel` / `AttractionTypeBroad` features instead of inventing numeric IDs.
- User and attraction joins have 100% coverage for the historical transaction table.

See `data/processed/cleaning_report.json` and `docs/DATA_DICTIONARY.md`.

## 3. Main EDA findings

- Average rating: **4.158 / 5**; median: **4.0**.
- 5-star ratings account for **45.22%** of transactions.
- Largest visit segment: **Couples** (40.85%).
- Peak recorded year: **2016**.
- Peak month by transaction volume: **August**.
- Most visited attraction: **Sacred Monkey Forest Sanctuary** (13,198 visits; avg. rating 4.267).
- `Waterbom Bali` has 6,429 visits and an average rating of 4.647 in the historical data.
- Asia is the highest-volume user continent in the supplied transaction set.

Static EDA charts are in `artifacts/figures/`; interactive equivalents appear in the Streamlit dashboard.

## 4. Predictive models

### Visit-mode classification

Three models are compared on a 70/15/15 train/validation/test split:

| Model | Validation Accuracy | Validation Weighted F1 |
|---|---:|---:|
| XGBoost | 0.502 | 0.455 |
| Random Forest | 0.448 | 0.453 |
| Logistic Regression | 0.325 | 0.352 |

**Selected model: XGBoost**. Held-out test metrics:

- Accuracy: **0.490**
- Weighted precision: **0.480**
- Weighted recall: **0.490**
- Weighted F1: **0.444**
- Macro F1: **0.294**

`Rating` is intentionally excluded from classification because it is not known before the trip and would create target-time leakage. `UserId` is also excluded so the model does not simply memorize identities.

### Rating regression

| Model | Validation RMSE | Validation MAE | Validation R² |
|---|---:|---:|---:|
| XGBoost | 0.905 | 0.709 | 0.131 |
| Random Forest | 0.913 | 0.713 | 0.116 |
| Ridge Regression | 0.922 | 0.723 | 0.098 |

**Selected model: XGBoost**. Held-out test metrics:

- RMSE: **0.902**
- MSE: **0.814**
- MAE: **0.710**
- R²: **0.136**

The moderate R² is reported honestly: the supplied demographic/attraction/time fields do not explain most individual rating variance.

## 5. Recommendation system

The project implements all three strategies:

- **Collaborative filtering:** item-item similarity over user ratings with user/item mean fallback.
- **Content-based filtering:** user profiles from attraction type/location attributes; can rank the full 1,698-site expanded catalog.
- **Hybrid:** blends content, collaborative estimates, and Bayesian-smoothed popularity.

Evaluation uses leave-one-out testing for users with at least two interactions. Ranking MAP@10 treats held-out ratings >=4 as relevant; the MAP calculation is run on a deterministic 1,000-user relevant sample for fast reproducibility.

- Collaborative rating RMSE: **1.094**
- Collaborative MAP@10: **0.095**
- Content MAP@10: **0.208**
- Hybrid MAP@10: **0.191**

Content-based recommendations are the strongest ranking method on this dataset, which is reasonable because 22,912 users have only one historical interaction and the collaborative matrix is sparse.

## 6. Streamlit application

The app has six sections:

1. **Executive Dashboard** - KPIs and interactive tourism visualizations.
2. **Visit Mode Prediction** - user/attraction inputs, predicted mode, class probabilities.
3. **Rating Prediction** - predicted attraction rating for a supplied profile and visit mode.
4. **Recommendations** - known-user content/collaborative/hybrid recommendations plus cold-start recommendations for new users.
5. **Model Evaluation** - comparisons and final regression/classification/recommendation metrics.
6. **SQL Analytics** - curated SQL queries for popular attractions, segments, types, continents, and year trends.

## 7. Project structure

```text
Tourism_Experience_Analytics_Complete_Project/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── Dockerfile
├── data/
│   ├── raw/                         # supplied Excel datasets
│   └── processed/                   # cleaned consolidated CSVs + cleaning report
├── database/tourism.db              # SQLite analytics database
├── src/
│   ├── config.py
│   ├── data_pipeline.py
│   ├── eda.py
│   ├── train_models.py
│   └── recommender.py
├── sql/analysis_queries.sql
├── artifacts/
│   ├── figures/                     # 8 EDA charts
│   ├── metrics/                     # model comparisons, reports, SQL outputs
│   └── models/                      # trained production artifacts
├── docs/
│   ├── FINAL_PROJECT_REPORT.docx
│   ├── FINAL_PROJECT_REPORT.md
│   ├── PROJECT_REQUIREMENTS_CHECKLIST.md
│   ├── DATA_DICTIONARY.md
│   ├── STREAMLIT_USER_GUIDE.md
│   └── PRESENTATION_NOTES.md
└── tests/
```

## 8. Setup and run

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The ZIP already contains generated cleaned data, SQLite DB, figures, metrics, and trained models. To rebuild everything from the raw Excel files:

```bash
python run_pipeline.py
```

Run automated tests:

```bash
pytest -q
```

Launch the application:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (normally `http://localhost:8501`).

## 9. Optional Docker run

```bash
docker build -t tourism-analytics .
docker run --rm -p 8501:8501 tourism-analytics
```

Open `http://localhost:8501`.

## 10. Important interpretation limitations

- The transaction history contains only 30 rated attractions, although the supplied updated catalog contains 1,698 sites. Therefore, direct collaborative evidence is limited to the 30 historical items; content-based methods are necessary for catalog expansion.
- The class distribution is imbalanced (Couples is the largest class and Business the smallest), so weighted and macro F1 are reported in addition to accuracy.
- The regression R² is modest, indicating that preference/satisfaction drivers not present in the dataset (price, trip purpose detail, amenities, review text, season/weather, party composition, etc.) would likely improve rating prediction.
- Historical tourism volume drops sharply after 2019 in the supplied data; this should not automatically be interpreted as a broad market trend without checking dataset coverage and external context.

## 11. Submission checklist

The requirement-by-requirement mapping is in `docs/PROJECT_REQUIREMENTS_CHECKLIST.md`. The project includes the cleaned dataset, source code, SQL database/query file, EDA outputs, trained models, evaluation files, Streamlit app, automated tests, report, user guide, and presentation notes.

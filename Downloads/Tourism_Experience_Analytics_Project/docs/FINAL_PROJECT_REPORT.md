# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

## Executive Summary

This project implements the complete tourism-analytics workflow specified in the project brief: data cleaning and preprocessing, exploratory data analysis (EDA), visualization, SQL analysis, rating regression, visit-mode classification, personalized recommendations, evaluation, and a Streamlit deployment layer.

The supplied transaction dataset contains 52,930 visits from 33,530 users and 30 attractions with historical ratings. The additional attraction catalog expands coverage to 1,698 attractions. The historical data is strongly positive in rating behavior (mean 4.158/5; 45.22% five-star ratings) and sparse at the user-item level: most users have only one historical interaction. This sparsity directly affects recommendation strategy and is why content-based recommendation is especially important.

XGBoost is the strongest tested predictive model for both supervised tasks. The held-out visit-mode classifier achieves 49.0% accuracy and 0.444 weighted F1. The held-out rating regressor achieves RMSE 0.902, MAE 0.710, and R² 0.136. These results show measurable signal but also meaningful uncertainty. For recommendations, content-based filtering delivers the strongest MAP@10 (0.208), followed by hybrid (0.191) and collaborative-only (0.095); collaborative rating RMSE is 1.094.

The final application exposes dashboards, predictions, recommendations, model comparisons, and SQL analytics in a user-facing Streamlit interface.

## 1. Business Problem and Objectives

Tourism platforms need to understand travel behavior, predict likely user behavior, estimate satisfaction, and recommend relevant attractions. The project addresses four practical business needs:

1. Personalized recommendations that increase engagement and customer retention.
2. Tourism analytics for attraction, region, time, and segment trends.
3. Visit-mode segmentation that can support targeted offers and campaign design.
4. Rating prediction that can serve as an expected-satisfaction indicator when prioritizing experiences.

The technical objectives are:
- Predict attraction rating on a 1-5 scale (regression).
- Predict visit mode: Business, Couples, Family, Friends, or Solo (classification).
- Rank personalized attractions using collaborative filtering and content-based filtering, with a hybrid method included as an optional enhancement.
- Deliver an interactive Streamlit application.

## 2. Dataset Assessment

The raw package contains transaction, user, city, country, region, continent, visit-mode, attraction-type, and attraction item tables, plus an updated attraction catalog.

### Historical modeling data
- Transactions: 52,930
- Users: 33,530
- Historical attractions: 30
- Rating range: 1-5
- Visit modes: Business, Couples, Family, Friends, Solo

### Expanded recommendation catalog
The updated item file contains 1,698 unique attraction IDs. Its `AttractionTypeId` column is not homogeneous: the original 30 attraction rows use numeric IDs, while 1,668 expanded rows contain textual labels such as Beach, Market, Museum, Park, and Temple. The pipeline does not overwrite or fabricate numeric IDs. It preserves the raw source value, parses numeric IDs where valid, and creates standardized text features (`AttractionTypeLabel` and `AttractionTypeBroad`) for recommendation modeling.

## 3. Data Cleaning and Preprocessing

The processing pipeline performs the following controls:

- Reads all supplied Excel tables using a single configuration-driven raw-data layer.
- Trims and normalizes lookup text, converting placeholder `-` values into `Unknown` for analysis labels.
- Fills four missing user `CityId` values with the dataset's existing placeholder key `0`.
- Removes exact transaction duplicates if present (none were present in this copy).
- Validates rating values to 1-5 and month values to 1-12.
- Maps numeric visit-mode IDs to human-readable labels.
- Joins user geography and attraction geography into one consolidated analytics table.
- Verifies 100% historical join coverage for user and attraction records.
- Creates a separate cleaned expanded attraction catalog for recommendation.
- Stores processed tables in CSV and SQLite form.

The final `tourism_merged_clean.csv` retains all 52,930 valid transactions.

## 4. Exploratory Data Analysis

### Rating behavior
The average rating is 4.158 and the median rating is 4.0. Five-star ratings represent 45.22% of all transactions, which indicates a strongly positive/left-skewed error environment for rating prediction. A naive mean prediction is therefore not sufficient, but the target also has limited variance.

### Visit-mode behavior
Couples is the largest observed visit mode at 40.85% of transactions. This class imbalance is why classification evaluation includes weighted and macro F1 in addition to accuracy.

### Time patterns
The highest transaction volume occurs in 2016 and the peak month is August. Historical volume drops sharply after 2019 in the supplied data. Because that may reflect collection coverage and not only tourism demand, the project does not claim that it is a universal market contraction.

### Attraction performance
Sacred Monkey Forest Sanctuary is the most visited historical attraction with 13,198 transactions and an average rating of 4.267. Waterbom Bali has 6,429 visits and an average rating of 4.647, combining high traffic with strong satisfaction.

The most visited attraction-type groups include Nature & Wildlife Areas, Beaches, Religious Sites, Water Parks, and Points of Interest & Landmarks.

## 5. SQL Analytics Layer

A SQLite database (`database/tourism.db`) is generated from the raw and processed tables. Curated SQL queries answer stakeholder questions such as:

- Which attractions have the highest visit volume?
- Which attractions combine strong ratings with sufficient visit volume?
- How do visit modes differ in volume and average rating?
- Which user continents and attraction types generate the most activity?
- What are the yearly and monthly patterns?
- How do user regions and visit modes interact as segments?

The full query library is stored in `sql/analysis_queries.sql` and several queries are directly exposed in the Streamlit app.

## 6. Classification: Visit-Mode Prediction

### Features
The classifier uses user continent/region/country/city, visit year/month, attraction identity/type/city/country. `Rating` is intentionally excluded because it would not be known before the visit. `UserId` is excluded to reduce identity memorization and improve the design for unseen users.

### Models compared
- Logistic Regression (one-hot encoded baseline)
- Random Forest
- XGBoost

A 70/15/15 train/validation/test split is used with target stratification and random state 42.

### Validation comparison
- XGBoost: accuracy 0.502; weighted F1 0.455
- Random Forest: accuracy 0.448; weighted F1 0.453
- Logistic Regression: accuracy 0.325; weighted F1 0.352

### Held-out test result (selected XGBoost)
- Accuracy: 0.490
- Weighted precision: 0.480
- Weighted recall: 0.490
- Weighted F1: 0.444
- Macro F1: 0.294

The macro F1 reveals that minority visit modes remain difficult. The model is useful for probability-based segmentation but should not be treated as deterministic labeling.

## 7. Regression: Attraction Rating Prediction

### Features
The regression model uses the same user/time/attraction context as classification and adds `VisitModeName`, which is a legitimate input when estimating expected satisfaction for a defined trip profile.

### Models compared
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

### Validation comparison
- XGBoost: RMSE 0.905; MAE 0.709; R² 0.131
- Random Forest: RMSE 0.913; MAE 0.713; R² 0.116
- Ridge Regression: RMSE 0.922; MAE 0.723; R² 0.098

### Held-out test result (selected XGBoost)
- RMSE: 0.902
- MSE: 0.814
- MAE: 0.710
- R²: 0.136

The R² value is modest. The model captures some systematic differences but most individual rating variance is not explained by the available demographic/time/location features. Additional variables such as price, party composition, trip purpose, amenities, review text, season/weather, and detailed traveler preferences would likely be valuable.

## 8. Recommendation System

Three recommendation approaches are implemented.

### Collaborative filtering
The collaborative component calculates item-item similarity from user ratings and predicts a candidate attraction using the user's mean rating plus similarity-weighted deviations. User/item/global means provide fallbacks.

### Content-based filtering
Each attraction is represented by type and location attributes. A known user's content profile is a rating-weighted combination of attractions they previously visited. This approach can rank all 1,698 attractions, including those without historical ratings.

### Hybrid system
The hybrid score blends content similarity, collaborative predicted rating, and Bayesian-smoothed popularity.

### Evaluation
Leave-one-out testing is performed for the 10,618 users with at least two interactions. Collaborative rating prediction is evaluated by RMSE. MAP@10 uses held-out interactions rated at least 4 as relevant; a deterministic 1,000-user relevant sample is used for fast repeatability.

Results:
- Collaborative RMSE: 1.094
- Collaborative MAP@10: 0.095
- Content MAP@10: 0.208
- Hybrid MAP@10: 0.191

Content-based filtering performs best on the ranking metric. This is consistent with the data structure: 22,912 users have only one interaction, and only 30 attractions have collaborative history. The expanded 1,698-item catalog makes content features essential for cold-start and catalog-expansion recommendations.

## 9. Streamlit Application

The application contains six user-facing sections:

1. Executive Dashboard - KPIs and interactive EDA.
2. Visit Mode Prediction - classification form plus class probabilities.
3. Rating Prediction - regression form with a 1-5 result.
4. Recommendations - known-user content/collaborative/hybrid ranking and cold-start preference ranking.
5. Model Evaluation - supervised model comparisons and recommendation metrics.
6. SQL Analytics - saved stakeholder queries with results.

The app loads the generated artifacts and can be started with:

`streamlit run app.py`

A Dockerfile is included for container deployment.

## 10. Business Insights and Recommendations

### Use content-based recommendation as the primary expansion strategy
Because historical attraction coverage is limited, content-based recommendation is currently both better on MAP@10 and capable of ranking the full catalog. Collaborative evidence should receive more weight only as multi-attraction histories grow.

### Use visit-mode probabilities for targeting, not hard decisions
The classifier provides useful segmentation signals, but accuracy and macro F1 show substantial uncertainty. Campaign systems should use predicted probabilities, business rules, and user-declared preferences together.

### Protect high-volume/high-rating experiences
Attractions such as Waterbom Bali combine strong demand and high satisfaction. These should be treated as anchor experiences in packages and recommendation surfaces, while still personalizing by user context.

### Investigate high-volume/lower-rating categories
High traffic does not guarantee the highest satisfaction. Categories with strong visit volume but lower average ratings can be prioritized for expectation setting, service improvement, or more precise audience matching.

### Improve data collection for rating prediction
The low-to-moderate R² suggests that current variables are insufficient to explain individual satisfaction. Richer trip and attraction context is the clearest path to stronger regression performance.

## 11. Limitations

- Only 30 attractions have transaction/rating histories.
- Most users have a single interaction, making collaborative filtering sparse.
- Classification classes are imbalanced.
- The updated catalog's type field is mixed-format and contains broad text labels not present as official numeric type IDs.
- The post-2019 transaction drop may be a coverage artifact.
- No price, weather, amenities, review text, party size, spend, or detailed trip-purpose fields are supplied.
- Metrics describe this dataset and split; they are not a guarantee of production performance.

## 12. Reproducibility and Quality Controls

The repository includes:
- `run_pipeline.py` to rebuild cleaning, EDA, models, and recommender.
- Modular `src/` code.
- Centralized paths and random state.
- Trained model artifacts.
- Processed CSVs and SQLite database.
- Model comparison files, prediction samples, confusion matrix, and feature-importance outputs.
- Automated tests (`pytest`) covering data integrity, model prediction, and recommendation output.
- Requirement checklist and Streamlit user guide.

The final test run completed successfully with 4 passing tests.

## 13. Conclusion

The project meets the requested end-to-end scope: it transforms the supplied tourism tables into a clean analytic dataset, derives stakeholder insights, compares multiple supervised models, evaluates three recommendation strategies, and deploys the results through Streamlit. The strongest practical recommendation is to prioritize content-based personalization while continuing to collect richer user histories and trip context, because the current collaborative matrix is sparse and the rating target is only partially explained by the available variables.

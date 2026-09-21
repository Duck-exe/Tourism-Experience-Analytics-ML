# Presentation / Demo Notes

## 1. Opening (30 seconds)
"My project is Tourism Experience Analytics. I built one integrated pipeline for tourism analytics, visit-mode classification, attraction-rating regression, and personalized recommendations. The deliverable also includes a SQLite analytics layer and a Streamlit application."

## 2. Dataset and cleaning (60 seconds)
Show `data/raw`, then `data/processed/cleaning_report.json`.

Mention:
- 52,930 transactions, 33,530 users.
- 30 historical transaction-linked attractions and an expanded catalog of 1,698 attractions.
- Four missing user city IDs were safely mapped to the dataset placeholder.
- All historical user and attraction joins resolve.
- The expanded item file mixes numeric and text attraction-type values. Instead of inventing type IDs, the pipeline preserves the raw value and creates standardized text labels.

## 3. EDA and SQL (60 seconds)
Open the Streamlit **Executive Dashboard** and **SQL Analytics** page.

Key findings:
- Mean rating 4.158.
- Couples is the largest visit mode.
- Peak year 2016; peak month August.
- Sacred Monkey Forest Sanctuary is the most visited historical attraction.
- Waterbom Bali has a high historical average rating (4.647) and high visit volume.

Explain that falling volume after 2019 may reflect data coverage, so it should not be claimed as a universal tourism trend without external validation.

## 4. Classification (60 seconds)
Open **Model Evaluation**, classification section.

"I compared Logistic Regression, Random Forest and XGBoost on the same 70/15/15 split. XGBoost performed best and achieved about 49% held-out accuracy and 0.444 weighted F1. I also report macro F1 because the classes are imbalanced. I deliberately excluded rating from the classifier because it would not be known before the visit."

Then demonstrate **Visit Mode Prediction**.

## 5. Regression (60 seconds)
"For rating prediction I compared Ridge, Random Forest and XGBoost. XGBoost was best with test RMSE about 0.902, MAE 0.710 and R² 0.136. The low R² is important: the available fields explain some patterns but much individual satisfaction remains unobserved."

Demonstrate **Rating Prediction**.

## 6. Recommendations (75 seconds)
"I implemented collaborative, content-based and optional hybrid recommendations. The catalog expansion matters because only 30 attractions have transaction histories, while the updated catalog contains 1,698. Content-based filtering can recommend unseen-catalog attractions using type and location attributes."

Metrics:
- collaborative RMSE 1.094
- collaborative MAP@10 0.095
- content MAP@10 0.208
- hybrid MAP@10 0.191

Demonstrate both **Known User** and **New / Cold-Start User**.

## 7. Business recommendations (45 seconds)
- Use predicted visit mode for campaign/package segmentation, but keep human/business rules around low-confidence predictions.
- Use content-based recommendations as the primary catalog-expansion method until more multi-attraction user history is collected.
- Prioritize attraction types with both strong satisfaction and visit volume, while separately investigating high-volume/lower-rating types.
- Collect richer context such as trip purpose, spend, party composition, amenities, reviews and season/weather to improve the rating model.

## 8. Close (20 seconds)
"The submission is reproducible through `python run_pipeline.py`, validated with automated tests, and deployable with `streamlit run app.py` or Docker. The report and checklist map every requirement to implementation evidence."

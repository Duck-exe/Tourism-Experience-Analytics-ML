import joblib
import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DIR
from src.train_models import CLASS_FEATURES, REG_FEATURES


def test_predictive_models_smoke():
    df = pd.read_csv(PROCESSED_DIR / "tourism_merged_clean.csv").head(3)
    clf = joblib.load(MODELS_DIR / "best_classification_model.joblib")
    reg = joblib.load(MODELS_DIR / "best_regression_model.joblib")
    assert len(clf.predict(df[CLASS_FEATURES])) == 3
    preds = reg.predict(df[REG_FEATURES])
    assert len(preds) == 3


def test_recommender_smoke():
    rec = joblib.load(MODELS_DIR / "hybrid_recommender.joblib")
    user_id = next(iter(rec.user_history.keys()))
    result = rec.recommend_known_user(user_id, k=5, strategy="content")
    assert len(result) == 5
    assert "Attraction" in result.columns

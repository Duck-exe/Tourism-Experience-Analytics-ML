import pandas as pd
from src.config import PROCESSED_DIR


def test_cleaned_dataset_integrity():
    df = pd.read_csv(PROCESSED_DIR / "tourism_merged_clean.csv")
    assert len(df) == 52930
    assert df["Rating"].between(1, 5).all()
    assert df["VisitMonth"].between(1, 12).all()
    assert df["UserId"].notna().all()
    assert df["AttractionId"].notna().all()
    assert df["VisitModeName"].notna().all()


def test_catalog_integrity():
    catalog = pd.read_csv(PROCESSED_DIR / "attraction_catalog_clean.csv")
    assert catalog["AttractionId"].nunique() == 1698
    assert not catalog["AttractionId"].duplicated().any()
    assert catalog["Attraction"].notna().all()
    assert catalog["AttractionTypeBroad"].notna().all()

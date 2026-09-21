from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR, DB_DIR

FILES = {
    "continent": "Continent.xlsx",
    "type": "Type.xlsx",
    "region": "Region.xlsx",
    "item": "Item.xlsx",
    "country": "Country.xlsx",
    "city": "City.xlsx",
    "mode": "Mode.xlsx",
    "user": "User.xlsx",
    "transaction": "Transaction.xlsx",
    "updated_item": "Additional_Data_for_Attraction_Sites/Updated_Item.xlsx",
}

BROAD_TYPE_MAP = {
    "Ancient Ruins": "Historic",
    "Ballets": "Culture",
    "Beaches": "Beach",
    "Beach": "Beach",
    "Caverns & Caves": "Nature",
    "Flea & Street Markets": "Market",
    "Market": "Market",
    "Historic Sites": "Historic",
    "History Museums": "Museum",
    "Speciality Museums": "Museum",
    "Museum": "Museum",
    "National Parks": "Park",
    "Park": "Park",
    "Nature & Wildlife Areas": "Nature",
    "Neighborhoods": "Neighborhood",
    "Points of Interest & Landmarks": "Landmark",
    "Religious Sites": "Religious",
    "Temple": "Religious",
    "Spas": "Spa",
    "Volcanos": "Nature",
    "Water Parks": "Water Park",
    "Waterfalls": "Nature",
}


def _read(name: str) -> pd.DataFrame:
    path = RAW_DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset: {path}")
    return pd.read_excel(path)


def load_raw_data() -> Dict[str, pd.DataFrame]:
    return {name: _read(name) for name in FILES}


def _clean_lookup_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"-": "Unknown", "": "Unknown", "<NA>": "Unknown"})
        .fillna("Unknown")
    )


def build_clean_datasets(raw: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    continent = raw["continent"].copy()
    region = raw["region"].copy()
    country = raw["country"].copy()
    city = raw["city"].copy()
    modes = raw["mode"].copy()
    types = raw["type"].copy()
    users = raw["user"].copy()
    tx = raw["transaction"].copy()
    item = raw["item"].copy()
    updated_item = raw["updated_item"].copy()

    # Normalize lookup text without destroying original IDs.
    continent["Continent"] = _clean_lookup_text(continent["Continent"])
    region["Region"] = _clean_lookup_text(region["Region"])
    country["Country"] = _clean_lookup_text(country["Country"])
    city["CityName"] = _clean_lookup_text(city["CityName"])
    modes["VisitMode"] = _clean_lookup_text(modes["VisitMode"])
    types["AttractionType"] = _clean_lookup_text(types["AttractionType"])

    # Four users have null CityId; the provided lookup uses 0 for unknown/placeholder.
    missing_user_city_before = int(users["CityId"].isna().sum())
    users["CityId"] = users["CityId"].fillna(0).astype("int64")

    # Remove exact duplicate rows if any and validate target ranges.
    tx_duplicates_before = int(tx.duplicated().sum())
    tx = tx.drop_duplicates().copy()
    tx = tx[
        tx["Rating"].between(1, 5)
        & tx["VisitMonth"].between(1, 12)
        & tx["VisitMode"].isin(modes.loc[modes["VisitModeId"] != 0, "VisitModeId"])
    ].copy()

    # User geography labels.
    user_city = city.rename(columns={"CityId": "UserCityId", "CityName": "UserCityName", "CountryId": "UserCityCountryId"})
    user_country = country.rename(columns={"CountryId": "UserCountryId", "Country": "UserCountry", "RegionId": "UserCountryRegionId"})
    user_region = region.rename(columns={"RegionId": "UserRegionId", "Region": "UserRegion", "ContinentId": "UserRegionContinentId"})
    user_cont = continent.rename(columns={"ContinentId": "UserContinentId", "Continent": "UserContinent"})

    users_labeled = (
        users.rename(columns={
            "ContinentId": "UserContinentId",
            "RegionId": "UserRegionId",
            "CountryId": "UserCountryId",
            "CityId": "UserCityId",
        })
        .merge(user_cont[["UserContinentId", "UserContinent"]], on="UserContinentId", how="left")
        .merge(user_region[["UserRegionId", "UserRegion"]], on="UserRegionId", how="left")
        .merge(user_country[["UserCountryId", "UserCountry"]], on="UserCountryId", how="left")
        .merge(user_city[["UserCityId", "UserCityName"]], on="UserCityId", how="left")
    )
    for c in ["UserContinent", "UserRegion", "UserCountry", "UserCityName"]:
        users_labeled[c] = users_labeled[c].fillna("Unknown")

    # Historical attraction table: IDs are clean and exactly cover all transaction AttractionId values.
    type_map = types.set_index("AttractionTypeId")["AttractionType"].to_dict()
    item["AttractionTypeLabel"] = item["AttractionTypeId"].map(type_map).fillna("Unknown")
    item["AttractionTypeBroad"] = item["AttractionTypeLabel"].map(BROAD_TYPE_MAP).fillna(item["AttractionTypeLabel"])

    attraction_city = city.rename(columns={"CityId": "AttractionCityId", "CityName": "AttractionCityName", "CountryId": "AttractionCountryId"})
    attraction_country = country.rename(columns={"CountryId": "AttractionCountryId", "Country": "AttractionCountry", "RegionId": "AttractionRegionId"})
    attraction_region = region.rename(columns={"RegionId": "AttractionRegionId", "Region": "AttractionRegion", "ContinentId": "AttractionContinentId"})
    attraction_cont = continent.rename(columns={"ContinentId": "AttractionContinentId", "Continent": "AttractionContinent"})

    item_labeled = (
        item.merge(attraction_city[["AttractionCityId", "AttractionCityName", "AttractionCountryId"]], on="AttractionCityId", how="left")
        .merge(attraction_country[["AttractionCountryId", "AttractionCountry", "AttractionRegionId"]], on="AttractionCountryId", how="left")
        .merge(attraction_region[["AttractionRegionId", "AttractionRegion", "AttractionContinentId"]], on="AttractionRegionId", how="left")
        .merge(attraction_cont[["AttractionContinentId", "AttractionContinent"]], on="AttractionContinentId", how="left")
    )
    for c in ["AttractionCityName", "AttractionCountry", "AttractionRegion", "AttractionContinent"]:
        item_labeled[c] = item_labeled[c].fillna("Unknown")

    mode_map = modes.set_index("VisitModeId")["VisitMode"].to_dict()
    tx["VisitModeName"] = tx["VisitMode"].map(mode_map).fillna("Unknown")

    merged = (
        tx.merge(users_labeled, on="UserId", how="left", validate="many_to_one")
        .merge(item_labeled, on="AttractionId", how="left", validate="many_to_one")
    )

    # Expanded catalog. Preserve mixed raw type values and derive a safe textual type label.
    updated_item = updated_item.drop_duplicates(subset=["AttractionId"], keep="first").copy()
    updated_item["AttractionTypeRaw"] = updated_item["AttractionTypeId"].astype("string").str.strip()
    numeric_type = pd.to_numeric(updated_item["AttractionTypeRaw"], errors="coerce")
    updated_item["AttractionTypeNumericId"] = numeric_type.astype("Int64")
    updated_item["AttractionTypeLabel"] = updated_item["AttractionTypeNumericId"].map(type_map)
    text_type_mask = updated_item["AttractionTypeLabel"].isna()
    updated_item.loc[text_type_mask, "AttractionTypeLabel"] = updated_item.loc[text_type_mask, "AttractionTypeRaw"]
    updated_item["AttractionTypeLabel"] = _clean_lookup_text(updated_item["AttractionTypeLabel"])
    updated_item["AttractionTypeBroad"] = updated_item["AttractionTypeLabel"].map(BROAD_TYPE_MAP).fillna(updated_item["AttractionTypeLabel"])
    expanded_catalog = (
        updated_item.drop(columns=["AttractionTypeId"])
        .merge(attraction_city[["AttractionCityId", "AttractionCityName", "AttractionCountryId"]], on="AttractionCityId", how="left")
        .merge(attraction_country[["AttractionCountryId", "AttractionCountry", "AttractionRegionId"]], on="AttractionCountryId", how="left")
        .merge(attraction_region[["AttractionRegionId", "AttractionRegion", "AttractionContinentId"]], on="AttractionRegionId", how="left")
        .merge(attraction_cont[["AttractionContinentId", "AttractionContinent"]], on="AttractionContinentId", how="left")
    )
    for c in ["AttractionCityName", "AttractionCountry", "AttractionRegion", "AttractionContinent"]:
        expanded_catalog[c] = expanded_catalog[c].fillna("Unknown")

    # Historical popularity/rating features are safe for recommendation display only, not predictive model inputs.
    attraction_stats = merged.groupby("AttractionId").agg(
        HistoricalVisitCount=("TransactionId", "count"),
        HistoricalAverageRating=("Rating", "mean"),
    ).reset_index()
    expanded_catalog = expanded_catalog.merge(attraction_stats, on="AttractionId", how="left")
    expanded_catalog["HistoricalVisitCount"] = expanded_catalog["HistoricalVisitCount"].fillna(0).astype(int)
    expanded_catalog["HistoricalAverageRating"] = expanded_catalog["HistoricalAverageRating"].fillna(merged["Rating"].mean())

    cleaning_report = {
        "raw_transaction_rows": int(len(raw["transaction"])),
        "clean_transaction_rows": int(len(merged)),
        "transaction_exact_duplicates_removed": tx_duplicates_before,
        "user_rows": int(len(users_labeled)),
        "user_missing_city_ids_filled_with_0": missing_user_city_before,
        "historical_attractions": int(item_labeled["AttractionId"].nunique()),
        "expanded_catalog_attractions": int(expanded_catalog["AttractionId"].nunique()),
        "expanded_catalog_mixed_text_type_rows": int(pd.to_numeric(raw["updated_item"]["AttractionTypeId"], errors="coerce").isna().sum()),
        "transaction_user_join_coverage": float(merged["UserContinent"].notna().mean()),
        "transaction_attraction_join_coverage": float(merged["Attraction"].notna().mean()),
        "rating_min": int(merged["Rating"].min()),
        "rating_max": int(merged["Rating"].max()),
    }
    return merged, expanded_catalog, cleaning_report


def save_outputs(raw: Dict[str, pd.DataFrame], merged: pd.DataFrame, expanded_catalog: pd.DataFrame, report: dict) -> None:
    merged.to_csv(PROCESSED_DIR / "tourism_merged_clean.csv", index=False)
    expanded_catalog.to_csv(PROCESSED_DIR / "attraction_catalog_clean.csv", index=False)
    with open(PROCESSED_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    db_path = DB_DIR / "tourism.db"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        for name, df in raw.items():
            safe_name = "updated_item_raw" if name == "updated_item" else f"{name}_raw"
            df.to_sql(safe_name, con, if_exists="replace", index=False)
        merged.to_sql("tourism_analytics", con, if_exists="replace", index=False)
        expanded_catalog.to_sql("attraction_catalog", con, if_exists="replace", index=False)
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_analytics_user ON tourism_analytics(UserId);
            CREATE INDEX IF NOT EXISTS idx_analytics_attraction ON tourism_analytics(AttractionId);
            CREATE INDEX IF NOT EXISTS idx_analytics_mode ON tourism_analytics(VisitMode);
            CREATE INDEX IF NOT EXISTS idx_catalog_attraction ON attraction_catalog(AttractionId);
            """
        )


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = load_raw_data()
    merged, catalog, report = build_clean_datasets(raw)
    save_outputs(raw, merged, catalog, report)
    print(json.dumps(report, indent=2))
    return merged, catalog, report


if __name__ == "__main__":
    run_pipeline()

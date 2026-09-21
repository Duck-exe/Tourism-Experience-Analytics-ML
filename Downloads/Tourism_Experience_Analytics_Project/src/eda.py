from __future__ import annotations

import json
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

from .config import PROCESSED_DIR, DB_DIR, FIGURES_DIR, METRICS_DIR


def _save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, horizontal: bool = False):
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        series.sort_values().plot(kind="barh", ax=ax)
    else:
        series.plot(kind="bar", ax=ax)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda() -> dict:
    df = pd.read_csv(PROCESSED_DIR / "tourism_merged_clean.csv")

    # Core distributions
    _save_bar(df["Rating"].value_counts().sort_index(), "Rating Distribution", "Rating", "Transactions", "01_rating_distribution.png")
    _save_bar(df["VisitModeName"].value_counts(), "Visit Mode Distribution", "Visit mode", "Transactions", "02_visit_mode_distribution.png")
    _save_bar(df["VisitYear"].value_counts().sort_index(), "Visits by Year", "Year", "Transactions", "03_visits_by_year.png")

    top_attr = df.groupby("Attraction").size().sort_values(ascending=False).head(12)
    _save_bar(top_attr, "Top Historical Attractions by Visit Count", "Visits", "Attraction", "04_top_attractions.png", horizontal=True)

    type_stats = df.groupby("AttractionTypeLabel").agg(visits=("TransactionId", "count"), avg_rating=("Rating", "mean")).sort_values("visits", ascending=False)
    _save_bar(type_stats.head(12)["visits"], "Most Visited Attraction Types", "Attraction type", "Visits", "05_attraction_type_popularity.png")

    continent_stats = df.groupby("UserContinent").size().sort_values(ascending=False)
    _save_bar(continent_stats, "Visits by User Continent", "User continent", "Visits", "06_user_continent_visits.png")

    # Monthly/year heatmap without seaborn.
    pivot = df.pivot_table(index="VisitYear", columns="VisitMonth", values="TransactionId", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Visit Volume by Year and Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(im, ax=ax, label="Visits")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_year_month_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    mode_rating = df.groupby("VisitModeName")["Rating"].mean().sort_values(ascending=False)
    _save_bar(mode_rating, "Average Rating by Visit Mode", "Visit mode", "Average rating", "08_avg_rating_by_visit_mode.png")

    # SQL-derived analysis tables.
    with sqlite3.connect(DB_DIR / "tourism.db") as con:
        top_attractions_sql = pd.read_sql_query(
            """SELECT AttractionId, Attraction, AttractionTypeLabel, COUNT(*) visits,
                      ROUND(AVG(Rating), 3) avg_rating
               FROM tourism_analytics GROUP BY AttractionId, Attraction, AttractionTypeLabel
               ORDER BY visits DESC LIMIT 15""", con)
        mode_sql = pd.read_sql_query(
            """SELECT VisitModeName, COUNT(*) visits, ROUND(AVG(Rating), 3) avg_rating
               FROM tourism_analytics GROUP BY VisitModeName ORDER BY visits DESC""", con)
        country_sql = pd.read_sql_query(
            """SELECT AttractionCountry, COUNT(*) visits, ROUND(AVG(Rating), 3) avg_rating
               FROM tourism_analytics GROUP BY AttractionCountry ORDER BY visits DESC""", con)
        type_sql = pd.read_sql_query(
            """SELECT AttractionTypeLabel, COUNT(*) visits, ROUND(AVG(Rating), 3) avg_rating
               FROM tourism_analytics GROUP BY AttractionTypeLabel ORDER BY visits DESC""", con)

    top_attractions_sql.to_csv(METRICS_DIR / "sql_top_attractions.csv", index=False)
    mode_sql.to_csv(METRICS_DIR / "sql_visit_modes.csv", index=False)
    country_sql.to_csv(METRICS_DIR / "sql_destination_countries.csv", index=False)
    type_sql.to_csv(METRICS_DIR / "sql_attraction_types.csv", index=False)

    rating_counts = df["Rating"].value_counts(normalize=True).sort_index()
    insights = {
        "transactions": int(len(df)),
        "users": int(df["UserId"].nunique()),
        "historical_attractions": int(df["AttractionId"].nunique()),
        "average_rating": round(float(df["Rating"].mean()), 3),
        "median_rating": float(df["Rating"].median()),
        "five_star_share": round(float(rating_counts.get(5, 0)), 4),
        "most_common_visit_mode": str(df["VisitModeName"].value_counts().idxmax()),
        "most_common_visit_mode_share": round(float(df["VisitModeName"].value_counts(normalize=True).max()), 4),
        "peak_year": int(df["VisitYear"].value_counts().idxmax()),
        "peak_month": int(df["VisitMonth"].value_counts().idxmax()),
        "most_visited_attraction": str(df["Attraction"].value_counts().idxmax()),
        "most_visited_attraction_visits": int(df["Attraction"].value_counts().max()),
        "highest_volume_user_continent": str(df["UserContinent"].value_counts().idxmax()),
        "highest_avg_rating_visit_mode": str(df.groupby("VisitModeName")["Rating"].mean().idxmax()),
    }
    with open(METRICS_DIR / "eda_insights.json", "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)
    print(json.dumps(insights, indent=2))
    return insights


if __name__ == "__main__":
    run_eda()

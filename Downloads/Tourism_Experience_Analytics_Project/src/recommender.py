from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from .config import PROCESSED_DIR, MODELS_DIR, METRICS_DIR

CONTENT_COLUMNS = [
    "AttractionTypeBroad",
    "AttractionCityName",
    "AttractionCountry",
    "AttractionRegion",
    "AttractionContinent",
]


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if not finite.any():
        return out
    lo, hi = np.nanmin(values[finite]), np.nanmax(values[finite])
    if hi - lo < 1e-12:
        out[finite] = 0.5
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return out


class HybridRecommender:
    """Collaborative + content + popularity recommender with cold-start support."""

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame):
        self.interactions = interactions[["UserId", "AttractionId", "Rating", "VisitModeName"]].copy()
        self.catalog = catalog.drop_duplicates("AttractionId").reset_index(drop=True).copy()
        self.catalog_index = {int(a): i for i, a in enumerate(self.catalog["AttractionId"].astype(int))}

        # Collaborative matrix is based only on attractions that have ratings.
        hist_ids = sorted(self.interactions["AttractionId"].unique().astype(int).tolist())
        self.hist_ids = np.array(hist_ids, dtype=int)
        self.hist_index = {a: i for i, a in enumerate(hist_ids)}
        users = sorted(self.interactions["UserId"].unique().astype(int).tolist())
        self.user_index = {u: i for i, u in enumerate(users)}

        rows = self.interactions["UserId"].map(self.user_index).to_numpy()
        cols = self.interactions["AttractionId"].map(self.hist_index).to_numpy()
        ratings = self.interactions["Rating"].astype(float).to_numpy()
        self.global_mean = float(np.mean(ratings))
        self.user_mean = self.interactions.groupby("UserId")["Rating"].mean().to_dict()
        self.item_mean = self.interactions.groupby("AttractionId")["Rating"].mean().to_dict()

        centered = ratings - self.interactions["UserId"].map(self.user_mean).to_numpy()
        mat = sparse.csr_matrix((centered, (rows, cols)), shape=(len(users), len(hist_ids)))
        self.item_similarity = cosine_similarity(mat.T)
        np.fill_diagonal(self.item_similarity, 0.0)

        self.user_history = {}
        for uid, grp in self.interactions.groupby("UserId"):
            # Compact Python tuples are much faster to serialize than thousands of tiny DataFrames.
            self.user_history[int(uid)] = list(zip(grp["AttractionId"].astype(int).tolist(), grp["Rating"].astype(float).tolist()))

        # Content space fit on the expanded catalog so new attractions can be recommended.
        content_frame = self.catalog[CONTENT_COLUMNS].fillna("Unknown").astype(str)
        self.content_encoder = OneHotEncoder(handle_unknown="ignore")
        self.catalog_matrix = self.content_encoder.fit_transform(content_frame)

        # Popularity uses Bayesian smoothing so low-count attractions do not dominate.
        stats = self.interactions.groupby("AttractionId").agg(count=("Rating", "size"), mean=("Rating", "mean"))
        prior = self.global_mean
        m = 20.0
        stats["bayes"] = (stats["count"] * stats["mean"] + m * prior) / (stats["count"] + m)
        self.popularity = stats["bayes"].to_dict()
        self.max_pop_count = max(1, int(stats["count"].max()))
        self.pop_count = stats["count"].to_dict()
        self.mode_item_mean = self.interactions.groupby(["VisitModeName", "AttractionId"])["Rating"].mean().to_dict()
        return self

    def _collab_score(self, user_id: int, attraction_id: int) -> float:
        user_id = int(user_id)
        attraction_id = int(attraction_id)
        umean = float(self.user_mean.get(user_id, self.global_mean))
        if user_id not in self.user_history or attraction_id not in self.hist_index:
            return float(self.item_mean.get(attraction_id, self.global_mean))
        history = self.user_history[user_id]
        target_idx = self.hist_index[attraction_id]
        sims, devs = [], []
        for other, rating in history:
            if other == attraction_id or other not in self.hist_index:
                continue
            sim = float(self.item_similarity[target_idx, self.hist_index[other]])
            if abs(sim) > 1e-12:
                sims.append(sim)
                devs.append(float(rating) - umean)
        if not sims:
            return float(0.6 * umean + 0.4 * self.item_mean.get(attraction_id, self.global_mean))
        pred = umean + float(np.dot(sims, devs) / (np.sum(np.abs(sims)) + 1e-12))
        return float(np.clip(pred, 1.0, 5.0))

    def _content_scores_for_user(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_history:
            return np.zeros(len(self.catalog), dtype=float)
        history = self.user_history[user_id]
        idxs, weights = [], []
        for aid, rating in history:
            if aid in self.catalog_index:
                idxs.append(self.catalog_index[aid])
                weights.append(max(float(rating) - 2.5, 0.25))
        if not idxs:
            return np.zeros(len(self.catalog), dtype=float)
        profile = sparse.csr_matrix(np.average(self.catalog_matrix[idxs].toarray(), axis=0, weights=np.array(weights)).reshape(1, -1))
        return cosine_similarity(profile, self.catalog_matrix).ravel()

    def recommend_known_user(self, user_id: int, k: int = 10, strategy: str = "hybrid") -> pd.DataFrame:
        user_id = int(user_id)
        content = self._content_scores_for_user(user_id)
        collab = np.array([
            self._collab_score(user_id, int(a)) if int(a) in self.hist_index else self.global_mean
            for a in self.catalog["AttractionId"]
        ])
        pop = np.array([
            self.popularity.get(int(a), self.global_mean) + 0.15 * math.log1p(self.pop_count.get(int(a), 0))
            for a in self.catalog["AttractionId"]
        ])
        if strategy == "collaborative":
            score = _minmax(collab)
        elif strategy == "content":
            score = _minmax(content)
        else:
            score = 0.52 * _minmax(content) + 0.30 * _minmax(collab) + 0.18 * _minmax(pop)

        visited = set()
        if user_id in self.user_history:
            visited = {aid for aid, _ in self.user_history[user_id]}
        eligible = ~self.catalog["AttractionId"].astype(int).isin(visited).to_numpy()
        if strategy == "collaborative":
            # Collaborative evidence only exists for historically rated attractions;
            # do not return arbitrary unseen-catalog items that all fall back to the global mean.
            eligible &= self.catalog["AttractionId"].astype(int).isin(set(self.hist_ids.tolist())).to_numpy()
        order = np.argsort(-score[eligible])[:k]
        result = self.catalog.loc[eligible].iloc[order].copy()
        result["RecommendationScore"] = score[eligible][order]
        result["PredictedRating"] = collab[eligible][order]
        cols = [
            "AttractionId", "Attraction", "AttractionTypeBroad", "AttractionCityName",
            "AttractionCountry", "AttractionAddress", "RecommendationScore", "PredictedRating",
            "HistoricalVisitCount", "HistoricalAverageRating"
        ]
        return result[[c for c in cols if c in result.columns]].reset_index(drop=True)

    def recommend_cold_start(
        self,
        attraction_type: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        preferred_visit_mode: Optional[str] = None,
        k: int = 10,
    ) -> pd.DataFrame:
        score = np.zeros(len(self.catalog), dtype=float)
        if attraction_type and attraction_type != "Any":
            score += 0.45 * (self.catalog["AttractionTypeBroad"].astype(str) == attraction_type).to_numpy(dtype=float)
        if country and country != "Any":
            score += 0.25 * (self.catalog["AttractionCountry"].astype(str) == country).to_numpy(dtype=float)
        if city and city != "Any":
            score += 0.20 * (self.catalog["AttractionCityName"].astype(str) == city).to_numpy(dtype=float)
        pop = np.array([self.popularity.get(int(a), self.global_mean) for a in self.catalog["AttractionId"]])
        score += 0.10 * _minmax(pop)
        if preferred_visit_mode and preferred_visit_mode != "Any":
            mode_scores = np.array([
                self.mode_item_mean.get((preferred_visit_mode, int(a)), self.global_mean)
                for a in self.catalog["AttractionId"]
            ])
            score += 0.10 * _minmax(mode_scores)
        order = np.argsort(-score)[:k]
        result = self.catalog.iloc[order].copy()
        result["RecommendationScore"] = score[order]
        cols = [
            "AttractionId", "Attraction", "AttractionTypeBroad", "AttractionCityName",
            "AttractionCountry", "AttractionAddress", "RecommendationScore",
            "HistoricalVisitCount", "HistoricalAverageRating"
        ]
        return result[[c for c in cols if c in result.columns]].reset_index(drop=True)


def _leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["UserId", "VisitYear", "VisitMonth", "TransactionId"])
    counts = ordered.groupby("UserId")["TransactionId"].transform("size")
    eligible = ordered[counts >= 2]
    test_idx = eligible.groupby("UserId").tail(1).index
    test = ordered.loc[test_idx].copy()
    train = ordered.drop(index=test_idx).copy()
    return train, test


def _evaluate_small_catalog(train: pd.DataFrame, test: pd.DataFrame, hist_catalog: pd.DataFrame) -> dict:
    rec = HybridRecommender().fit(train, hist_catalog)
    preds = [rec._collab_score(int(r.UserId), int(r.AttractionId)) for r in test.itertuples()]
    rmse = math.sqrt(mean_squared_error(test["Rating"].astype(float), preds))

    # MAP@10 for relevant held-out interactions (rating >= 4).
    relevant = test[test["Rating"] >= 4].copy()
    ap_collab, ap_content, ap_hybrid = [], [], []
    # Deterministic sample cap keeps reruns fast while still evaluating thousands of users.
    if len(relevant) > 1000:
        relevant = relevant.sample(1000, random_state=42)
    for r in relevant.itertuples():
        target = int(r.AttractionId)
        for strategy, bucket in [("collaborative", ap_collab), ("content", ap_content), ("hybrid", ap_hybrid)]:
            top = rec.recommend_known_user(int(r.UserId), k=10, strategy=strategy)
            ids = top["AttractionId"].astype(int).tolist()
            bucket.append(1.0 / (ids.index(target) + 1) if target in ids else 0.0)
    return {
        "collaborative_rmse": float(rmse),
        "map_at_10_collaborative": float(np.mean(ap_collab)) if ap_collab else 0.0,
        "map_at_10_content": float(np.mean(ap_content)) if ap_content else 0.0,
        "map_at_10_hybrid": float(np.mean(ap_hybrid)) if ap_hybrid else 0.0,
        "leave_one_out_test_users": int(test["UserId"].nunique()),
        "map_evaluated_relevant_users": int(len(relevant)),
        "relevance_threshold": "held-out Rating >= 4",
    }


def train_recommender() -> dict:
    interactions = pd.read_csv(PROCESSED_DIR / "tourism_merged_clean.csv")
    catalog = pd.read_csv(PROCESSED_DIR / "attraction_catalog_clean.csv")
    historical_catalog = catalog[catalog["AttractionId"].isin(interactions["AttractionId"].unique())].copy()

    train, test = _leave_one_out_split(interactions)
    metrics = _evaluate_small_catalog(train, test, historical_catalog)

    final = HybridRecommender().fit(interactions, catalog)
    joblib.dump(final, MODELS_DIR / "hybrid_recommender.joblib", compress=1)
    with open(METRICS_DIR / "recommendation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Example outputs for reproducibility.
    example_user = int(interactions.groupby("UserId").size().sort_values(ascending=False).index[0])
    final.recommend_known_user(example_user, k=10, strategy="hybrid").to_csv(METRICS_DIR / "recommendation_example_known_user.csv", index=False)
    final.recommend_cold_start(k=10).to_csv(METRICS_DIR / "recommendation_example_cold_start.csv", index=False)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_recommender()

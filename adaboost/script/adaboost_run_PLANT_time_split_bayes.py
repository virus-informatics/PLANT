#!/usr/bin/env python
"""AdaBoost baseline for PLANT time/season splitting.

This script is intended to compare the original AAindex + AdaBoost baseline against
PLANT under the same paired-data cleaning and split strategy used by the PLANT
training script:

1. Clean the paired HI dataset with the same required columns and sequence filters.
2. In season mode, split rows with season <= cutoff into the past set and rows with
   later seasons into the future set.
3. Split the past set into train_val/test using StratifiedGroupKFold with
   groups=virus and stratification labels=virus_collection_year.
4. Split train_val into train/validation with the same strategy.
5. Run BayesSearchCV on the training split, then evaluate validation/test/future.

Example:
    python Training/adaboost_run_PLANT_time_split_bayes.py \
        --prefix adaboost_aaindex \
        --directory . \
        --paired-csv data/WHO_GISAID_dataset_final_strict_score_250124.csv \
        --aaindex-csv data/AAindex_GIAG010101_table.csv \
        --split-mode season \
        --cutoff-season NH2022 \
        --n-iter 50 \
        --n-jobs 14

Run all possible cutoff seasons:
    python Training/adaboost_run_PLANT_time_split_bayes.py \
        --prefix adaboost_aaindex_all \
        --directory . \
        --paired-csv data/WHO_GISAID_dataset_final_strict_score_250124.csv \
        --aaindex-csv data/AAindex_GIAG010101_table.csv \
        --all-cutoff-seasons
"""

from __future__ import annotations

import argparse
import calendar
import json
import pickle
import random
import re
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import scipy.stats
from scipy import sparse
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


SEED_DEFAULT = 100
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
PREDICTION_COLUMNS = ["predicted_dist", "predicted_dist_censor_cap"]

_SEASON_RE_NH_SH_FIRST = re.compile(r"^\s*(NH|SH)\s*[-_/ ]?\s*(\d{4})\s*$", re.IGNORECASE)
_SEASON_RE_YEAR_FIRST = re.compile(r"^\s*(\d{4})\s*[-_/ ]?\s*(NH|SH)\s*$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an AAindex + AdaBoost baseline using PLANT-compatible splits."
    )
    parser.add_argument("--prefix", required=True, help="Output prefix/version name.")
    parser.add_argument(
        "--directory",
        default=".",
        help="Repository or project directory. Outputs are written under Season_based_split_performance/.",
    )
    parser.add_argument(
        "--paired-csv",
        default=None,
        help=(
            "Paired antigenic-distance CSV. Default: "
            "data/WHO_GISAID_dataset_final_strict_score_250124.csv under --directory."
        ),
    )
    parser.add_argument(
        "--aaindex-csv",
        default=None,
        help=(
            "20x20 AAindex table CSV. Default search order: "
            "data/AAindex_GIAG010101_table.csv, then AAindex_GIAG010101_table.csv under --directory."
        ),
    )
    parser.add_argument("--max-length", "--max_length", dest="max_length", default=329, type=int)
    parser.add_argument("--random-seed", "--random_seed", dest="random_seed", default=SEED_DEFAULT, type=int)
    parser.add_argument(
        "--split-mode",
        "--split_mode",
        dest="split_mode",
        choices=["season", "full"],
        default="season",
        help=(
            "'season': rows with season <= cutoff are split into train/val/test; later seasons are future. "
            "'full': all rows are split into train/val/test and no future set is created."
        ),
    )
    parser.add_argument(
        "--cutoff-season",
        "--cutoff_season",
        dest="cutoff_season",
        default="NH2022",
        help="Cutoff season used when --split-mode season. Examples: NH2014, SH2014, 2014NH.",
    )
    parser.add_argument(
        "--cutoff-seasons",
        "--cutoff_seasons",
        dest="cutoff_seasons",
        default=None,
        help="Comma-separated cutoff seasons to run, e.g. NH2014,SH2014,NH2015.",
    )
    parser.add_argument(
        "--all-cutoff-seasons",
        "--all_cutoff_seasons",
        dest="all_cutoff_seasons",
        action="store_true",
        help="Run every parsable cutoff season except the last season in the cleaned data.",
    )
    parser.add_argument(
        "--season-col",
        "--season_col",
        dest="season_col",
        default="season",
        help="Column containing season labels such as NH2014 or SH2014.",
    )
    parser.add_argument(
        "--metadata-features",
        "--metadata_features",
        dest="metadata_features",
        choices=["plant", "plant_with_date", "passage_only", "none"],
        default="plant",
        help=(
            "Categorical metadata appended to AAindex features. "
            "'plant' uses virus/reference/virus_passage/reference_passage, matching the PLANT metadata encoders. "
            "'plant_with_date' additionally includes date."
        ),
    )
    parser.add_argument(
        "--no-density-weights",
        "--no_density_weights",
        dest="use_density_weights",
        action="store_false",
        help="Disable PLANT-style score-density sample weights for AdaBoost fitting.",
    )
    parser.set_defaults(use_density_weights=True)
    parser.add_argument(
        "--tune-on",
        choices=["train", "train_val"],
        default="train",
        help=(
            "Data used for BayesSearchCV. Default 'train' mirrors PLANT's explicit train split; "
            "'train_val' uses the full 80%% past split for baseline model selection."
        ),
    )
    parser.add_argument(
        "--final-fit-on",
        choices=["train", "train_val", "same_as_tune"],
        default="same_as_tune",
        help="Data used to refit the best AdaBoost model before evaluation.",
    )
    parser.add_argument("--n-iter", "--n_iter", dest="n_iter", default=50, type=int)
    parser.add_argument("--cv", default=5, type=int, help="Inner CV folds for BayesSearchCV.")
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", default=1, type=int)
    parser.add_argument("--verbose", default=2, type=int)
    parser.add_argument(
        "--score-transform",
        "--score_transform",
        dest="score_transform",
        choices=["none", "clip_0_1"],
        default="none",
        help="Optional transform of score before training/evaluation. Default preserves PLANT's score column.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_season_label(season_value) -> tuple[str, int]:
    """Parse season label and return normalized label plus sortable order.

    Convention: ... SH2013 < NH2014 < SH2014 < NH2015 ...
    NHYYYY -> YYYY * 2, SHYYYY -> YYYY * 2 + 1.
    """
    if pd.isna(season_value):
        raise ValueError("Season label is missing.")

    raw = str(season_value).strip().upper()
    match = _SEASON_RE_NH_SH_FIRST.match(raw)
    if match:
        hemisphere, year_str = match.groups()
    else:
        match = _SEASON_RE_YEAR_FIRST.match(raw)
        if not match:
            raise ValueError(
                f"Could not parse season label {season_value!r}. "
                "Expected labels such as NH2014, SH2014, 2014NH, or 2014SH."
            )
        year_str, hemisphere = match.groups()
        hemisphere = hemisphere.upper()

    year = int(year_str)
    normalized = f"{hemisphere}{year:04d}"
    order = year * 2 + (1 if hemisphere == "SH" else 0)
    return normalized, order


def season_order_to_label(order: int) -> str:
    order = int(order)
    if order % 2 == 0:
        return f"NH{order // 2:04d}"
    return f"SH{(order - 1) // 2:04d}"


def add_season_order_columns(df: pd.DataFrame, *, season_col: str = "season") -> pd.DataFrame:
    out = df.copy()
    parsed_labels: list[str | None] = []
    parsed_orders: list[float] = []
    invalid_values = []

    for value in out[season_col]:
        try:
            normalized, order = parse_season_label(value)
        except ValueError:
            parsed_labels.append(None)
            parsed_orders.append(np.nan)
            invalid_values.append(value)
        else:
            parsed_labels.append(normalized)
            parsed_orders.append(order)

    out["season_normalized"] = parsed_labels
    out["season_order"] = parsed_orders

    invalid = out["season_order"].isna()
    if invalid.any():
        examples = sorted({str(v) for v in invalid_values})[:10]
        print(
            "Dropping rows with unparseable season values for season split: "
            f"{int(invalid.sum())}. Examples: {examples}"
        )
        out = out.loc[~invalid].copy()

    out["season_order"] = out["season_order"].astype(int)
    return out


def split_past_future_by_cutoff_season(
    df: pd.DataFrame,
    *,
    cutoff_season: str,
    season_col: str = "season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_normalized, cutoff_order = parse_season_label(cutoff_season)
    out = add_season_order_columns(df, season_col=season_col)

    past = out.loc[out["season_order"] <= cutoff_order].copy()
    future = out.loc[out["season_order"] > cutoff_order].copy()

    if past.empty:
        raise ValueError(f"No rows assigned to past set with cutoff_season={cutoff_normalized!r}.")
    if future.empty:
        raise ValueError(f"No rows assigned to future set with cutoff_season={cutoff_normalized!r}.")

    return past.reset_index(drop=True), future.reset_index(drop=True)


def clean_paired_dataset(
    path: Path,
    max_length: int,
    seed: int,
    *,
    season_col: str = "season",
    score_transform: str = "none",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if season_col != "season":
        if season_col not in df.columns:
            raise ValueError(f"Missing requested season column in {path}: {season_col!r}")
        df = df.rename(columns={season_col: "season"})

    required_cols = [
        "date",
        "season",
        "virus",
        "reference",
        "virus_passage",
        "reference_passage",
        "score",
        "censor",
        "virus_seq",
        "reference_seq",
        "virus_collection_date",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for col in ["virus_seq", "reference_seq"]:
        df[col] = df[col].astype(str)
        df = df[~df[col].str.contains("X", regex=False, na=False)]
        df = df[~df[col].str.contains("B", regex=False, na=False)]
        df = df[~df[col].str.contains("*", regex=False, na=False)]
    if "pair_strict" in df.columns:
        df = df[~df["pair_strict"].astype(str).str.contains("|", regex=False, na=False)]

    df = df[df["virus_seq"].str.len() == max_length]
    df = df[df["reference_seq"].str.len() == max_length]

    # Same blacklist logic as the PLANT script: drop date/reference/passage groups whose self-distance is suspiciously high.
    blacklist_df = df[
        (df["virus"] == df["reference"]) & (df["score"].astype(float) >= 0.25)
    ][["date", "reference", "reference_passage"]].drop_duplicates()
    df = (
        df.merge(
            blacklist_df,
            on=["date", "reference", "reference_passage"],
            how="left",
            indicator=True,
        )
        .query('_merge == "left_only"')
        .drop(columns=["_merge"])
    )

    selected = df[required_cols].dropna().copy()
    selected["score"] = selected["score"].astype(float)
    selected["censor"] = selected["censor"].astype(float)
    if score_transform == "clip_0_1":
        selected["score"] = selected["score"].clip(0.0, 1.0)

    selected = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    selected["virus_collection_year"] = pd.to_datetime(
        selected["virus_collection_date"], errors="coerce"
    ).dt.year
    selected = selected.dropna(subset=["virus_collection_year"]).copy()
    selected["virus_collection_year"] = selected["virus_collection_year"].astype(int)

    for col in ["date", "virus", "reference", "virus_passage", "reference_passage"]:
        selected[f"{col}_category"] = selected[col].astype("category").cat.codes

    return selected.reset_index(drop=True)


def split_dataframe_with_stratified_group_kfold(
    df: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_splits = max(2, int(round(1 / (1 - train_ratio))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["virus"]
    stratify_labels = df["virus_collection_year"]
    for train_idx, val_idx in skf.split(df, stratify_labels, groups):
        return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
    raise RuntimeError("StratifiedGroupKFold did not produce a split.")


def add_density_weights(train_df: pd.DataFrame, *, num_bins: int = 9, max_weight: float = 5.0) -> pd.DataFrame:
    train_df = train_df.copy()
    bin_edges = np.linspace(0, 1, num_bins + 1)
    unique_group_means = (
        train_df.groupby(["virus", "reference", "virus_passage", "reference_passage"])["score"]
        .mean()
        .reset_index()
    )
    unique_group_means["bin"] = pd.cut(
        unique_group_means["score"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )
    bin_counts = unique_group_means["bin"].value_counts().sort_index()
    total_samples = len(unique_group_means)
    bin_weights = {bin_idx: total_samples / count for bin_idx, count in bin_counts.items()}
    unique_group_means["weight"] = unique_group_means["bin"].map(bin_weights)
    train_df = train_df.merge(
        unique_group_means[
            ["virus", "reference", "virus_passage", "reference_passage", "bin", "weight"]
        ],
        on=["virus", "reference", "virus_passage", "reference_passage"],
        how="left",
    )
    min_weight = train_df["weight"].min()
    train_df["weight"] = (train_df["weight"] / min_weight).clip(upper=max_weight)
    return train_df


def make_one_hot_encoder(values) -> OneHotEncoder:
    values = np.asarray(values).reshape(-1, 1)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    return encoder.fit(values)


def resolve_aaindex_path(storage_path: Path, aaindex_arg: str | None) -> Path:
    if aaindex_arg:
        path = Path(aaindex_arg)
        if path.exists():
            return path
        candidate = storage_path / aaindex_arg
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"AAindex CSV not found: {aaindex_arg}")

    candidates = [
        storage_path / "data" / "AAindex_GIAG010101_table.csv",
        storage_path / "AAindex_GIAG010101_table.csv",
        Path("AAindex_GIAG010101_table.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "AAindex CSV not found. Pass --aaindex-csv, e.g. data/AAindex_GIAG010101_table.csv"
    )


def load_aaindex(path: Path) -> np.ndarray:
    matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if matrix.shape[0] < 20 or matrix.shape[1] < 20:
        raise ValueError(f"AAindex matrix must be at least 20x20, got {matrix.shape} from {path}")
    return matrix[:20, :20]


def aaindex_pair_features(df: pd.DataFrame, aaindex_matrix: np.ndarray, *, max_length: int) -> np.ndarray:
    aa_to_idx = {aa: idx for idx, aa in enumerate(AA_ORDER)}
    X = np.empty((len(df), max_length), dtype=np.float32)

    for row_idx, (virus_seq, reference_seq) in enumerate(zip(df["virus_seq"], df["reference_seq"])):
        if len(virus_seq) != max_length or len(reference_seq) != max_length:
            raise ValueError(
                f"Unexpected sequence length at row {row_idx}: "
                f"virus={len(virus_seq)}, reference={len(reference_seq)}, expected={max_length}"
            )
        for pos, (v, s) in enumerate(zip(virus_seq, reference_seq)):
            try:
                vi = aa_to_idx[v]
                si = aa_to_idx[s]
            except KeyError as exc:
                raise ValueError(
                    f"Invalid amino acid at row={row_idx}, pos={pos + 1}: virus={v!r}, reference={s!r}"
                ) from exc
            X[row_idx, pos] = aaindex_matrix[vi, vi] + aaindex_matrix[si, si] - 2.0 * aaindex_matrix[si, vi]
    return X


def metadata_columns_for_mode(mode: str) -> list[str]:
    if mode == "none":
        return []
    if mode == "passage_only":
        return ["virus_passage", "reference_passage"]
    if mode == "plant":
        return ["virus", "reference", "virus_passage", "reference_passage"]
    if mode == "plant_with_date":
        return ["date", "virus", "reference", "virus_passage", "reference_passage"]
    raise ValueError(f"Unknown metadata feature mode: {mode!r}")


def fit_metadata_encoders(train_df: pd.DataFrame, metadata_cols: Iterable[str]) -> dict[str, OneHotEncoder]:
    encoders: dict[str, OneHotEncoder] = {}
    for col in metadata_cols:
        encoders[col] = make_one_hot_encoder(train_df[col].astype(str))
    return encoders


def transform_metadata(df: pd.DataFrame, encoders: dict[str, OneHotEncoder]) -> sparse.csr_matrix:
    if not encoders:
        return sparse.csr_matrix((len(df), 0), dtype=np.float32)
    parts = []
    for col, encoder in encoders.items():
        parts.append(encoder.transform(df[col].astype(str).to_numpy().reshape(-1, 1)))
    return sparse.hstack(parts, format="csr", dtype=np.float32)


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    aaindex_matrix: np.ndarray,
    metadata_encoders: dict[str, OneHotEncoder],
    max_length: int,
) -> sparse.csr_matrix:
    seq_X = sparse.csr_matrix(aaindex_pair_features(df, aaindex_matrix, max_length=max_length))
    meta_X = transform_metadata(df, metadata_encoders)
    return sparse.hstack([seq_X, meta_X], format="csr", dtype=np.float32)


def _safe_correlation(x, y, *, method: str) -> tuple[float, float, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[ok]
    y_arr = y_arr[ok]
    n = int(x_arr.size)
    if n < 3 or np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return float("nan"), float("nan"), n
    if method == "pearson":
        corr, p_value = scipy.stats.pearsonr(x_arr, y_arr)
    elif method == "spearman":
        corr, p_value = scipy.stats.spearmanr(x_arr, y_arr)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(corr), float(p_value), n


def _safe_error(y_true, y_pred, *, metric: str) -> tuple[float, int]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[ok]
    y_pred_arr = y_pred_arr[ok]
    n = int(y_true_arr.size)
    if n == 0:
        return float("nan"), n
    residual = y_pred_arr - y_true_arr
    if metric == "mae":
        value = np.mean(np.abs(residual))
    elif metric == "rmse":
        value = np.sqrt(np.mean(residual**2))
    else:
        raise ValueError(f"Unknown error metric: {metric}")
    return float(value), n


def _safe_lower_bound_violation_rate(score, censor, prediction) -> tuple[float, int]:
    score_arr = np.asarray(score, dtype=float)
    censor_arr = np.asarray(censor, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    ok = np.isfinite(score_arr) & np.isfinite(censor_arr) & np.isfinite(pred_arr)
    censored = ok & (censor_arr == 1)
    n = int(censored.sum())
    if n == 0:
        return float("nan"), n
    violation_rate = np.mean(pred_arr[censored] < score_arr[censored])
    return float(violation_rate), n


def apply_censor_cap(df: pd.DataFrame, censor_col: str, predicted_col: str, score_col: str, output_col: str) -> None:
    # PLANT-compatible censor cap behavior.
    df[output_col] = np.where(
        (df[censor_col].astype(float) == 1) & (df[predicted_col] > df[score_col]),
        df[score_col],
        df[predicted_col],
    )


def compute_prediction_metrics(score, censor, prediction_columns: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    score_arr = np.asarray(score, dtype=float)
    censor_arr = np.asarray(censor, dtype=float)

    for name, pred in prediction_columns.items():
        pred_arr = np.asarray(pred, dtype=float)
        pearson_r, pearson_p, pearson_n = _safe_correlation(score_arr, pred_arr, method="pearson")
        spearman_r, spearman_p, spearman_n = _safe_correlation(score_arr, pred_arr, method="spearman")
        mae, mae_n = _safe_error(score_arr, pred_arr, metric="mae")
        rmse, rmse_n = _safe_error(score_arr, pred_arr, metric="rmse")

        uncensored = censor_arr == 0
        uncensored_mae, uncensored_mae_n = _safe_error(
            score_arr[uncensored], pred_arr[uncensored], metric="mae"
        )
        uncensored_rmse, uncensored_rmse_n = _safe_error(
            score_arr[uncensored], pred_arr[uncensored], metric="rmse"
        )
        violation_rate, violation_n = _safe_lower_bound_violation_rate(score_arr, censor_arr, pred_arr)

        metrics[f"pearson_{name}"] = pearson_r
        metrics[f"pearson_p_{name}"] = pearson_p
        metrics[f"pearson_n_{name}"] = pearson_n
        metrics[f"spearman_{name}"] = spearman_r
        metrics[f"spearman_p_{name}"] = spearman_p
        metrics[f"spearman_n_{name}"] = spearman_n
        metrics[f"mae_{name}"] = mae
        metrics[f"mae_n_{name}"] = mae_n
        metrics[f"rmse_{name}"] = rmse
        metrics[f"rmse_n_{name}"] = rmse_n
        metrics[f"uncensored_mae_{name}"] = uncensored_mae
        metrics[f"uncensored_mae_n_{name}"] = uncensored_mae_n
        metrics[f"uncensored_rmse_{name}"] = uncensored_rmse
        metrics[f"uncensored_rmse_n_{name}"] = uncensored_rmse_n
        metrics[f"censored_lower_bound_violation_rate_{name}"] = violation_rate
        metrics[f"censored_lower_bound_violation_n_{name}"] = violation_n

    if "predicted_dist_censor_cap" in prediction_columns:
        metrics["pearson_censor_cap"] = metrics["pearson_predicted_dist_censor_cap"]
        metrics["spearman_censor_cap"] = metrics["spearman_predicted_dist_censor_cap"]
        metrics["mae_censor_cap"] = metrics["mae_predicted_dist_censor_cap"]
        metrics["rmse_censor_cap"] = metrics["rmse_predicted_dist_censor_cap"]

    return metrics


def compute_prediction_metrics_from_dataframe(df: pd.DataFrame) -> dict[str, float]:
    available_predictions = {col: df[col].to_numpy() for col in PREDICTION_COLUMNS if col in df.columns}
    return compute_prediction_metrics(df["score"].to_numpy(), df["censor"].to_numpy(), available_predictions)


def write_prediction_metrics(df: pd.DataFrame, output_path: Path) -> None:
    metrics = compute_prediction_metrics_from_dataframe(df)
    lines = []
    for col in PREDICTION_COLUMNS:
        if f"pearson_{col}" not in metrics:
            continue
        lines.extend(
            [
                f"[{col}]",
                f"pearson_r\t{metrics[f'pearson_{col}']:.8g}",
                f"pearson_p\t{metrics[f'pearson_p_{col}']:.8g}",
                f"pearson_n\t{int(metrics[f'pearson_n_{col}'])}",
                f"spearman_r\t{metrics[f'spearman_{col}']:.8g}",
                f"spearman_p\t{metrics[f'spearman_p_{col}']:.8g}",
                f"spearman_n\t{int(metrics[f'spearman_n_{col}'])}",
                f"mae\t{metrics[f'mae_{col}']:.8g}",
                f"mae_n\t{int(metrics[f'mae_n_{col}'])}",
                f"rmse\t{metrics[f'rmse_{col}']:.8g}",
                f"rmse_n\t{int(metrics[f'rmse_n_{col}'])}",
                f"uncensored_mae\t{metrics[f'uncensored_mae_{col}']:.8g}",
                f"uncensored_mae_n\t{int(metrics[f'uncensored_mae_n_{col}'])}",
                f"uncensored_rmse\t{metrics[f'uncensored_rmse_{col}']:.8g}",
                f"uncensored_rmse_n\t{int(metrics[f'uncensored_rmse_n_{col}'])}",
                "censored_lower_bound_violation_rate\t"
                f"{metrics[f'censored_lower_bound_violation_rate_{col}']:.8g}",
                f"censored_lower_bound_violation_n\t{int(metrics[f'censored_lower_bound_violation_n_{col}'])}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines))


def _json_safe_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compute_prediction_metrics_by_season(df: pd.DataFrame, *, season_col: str = "season_normalized") -> pd.DataFrame:
    if season_col not in df.columns:
        raise ValueError(f"Missing season column for per-season metrics: {season_col!r}")

    rows = []
    sort_cols = [season_col]
    if "season_order" in df.columns:
        sort_cols = ["season_order", season_col]

    for season, group in df.sort_values(sort_cols).groupby(season_col, sort=False):
        metrics = compute_prediction_metrics_from_dataframe(group)
        row = {"season": season, "n": int(len(group))}
        if "season_order" in group.columns:
            row["season_order"] = int(group["season_order"].iloc[0])
        row.update(metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    if not metrics_df.empty and "season_order" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(["season_order", "season"]).reset_index(drop=True)
    return metrics_df


def write_future_metrics_by_season(df: pd.DataFrame, outputs_path: Path, *, season_col: str = "season_normalized") -> None:
    metrics_df = compute_prediction_metrics_by_season(df, season_col=season_col)
    csv_path = outputs_path / "future_prediction_metrics_by_season.csv"
    metrics_df.to_csv(csv_path, index=False)

    json_ready = {}
    for _, row in metrics_df.iterrows():
        season = row["season"]
        json_ready[str(season)] = {
            str(key): _json_safe_value(value) for key, value in row.items() if key != "season"
        }
    (outputs_path / "future_prediction_metrics_by_season.json").write_text(
        json.dumps(json_ready, indent=2, sort_keys=True)
    )

    lines = []
    for _, row in metrics_df.iterrows():
        lines.append(f"[{row['season']}] n={int(row['n'])}")
        for col in PREDICTION_COLUMNS:
            if f"pearson_{col}" not in row:
                continue
            lines.extend(
                [
                    f"  {col}",
                    f"    pearson_r\t{row[f'pearson_{col}']:.8g}",
                    f"    pearson_n\t{int(row[f'pearson_n_{col}'])}",
                    f"    spearman_r\t{row[f'spearman_{col}']:.8g}",
                    f"    spearman_n\t{int(row[f'spearman_n_{col}'])}",
                    f"    mae\t{row[f'mae_{col}']:.8g}",
                    f"    rmse\t{row[f'rmse_{col}']:.8g}",
                ]
            )
        lines.append("")
    (outputs_path / "future_prediction_metrics_by_season.txt").write_text("\n".join(lines))


def predict_and_save_split(
    model: AdaBoostRegressor,
    X: sparse.csr_matrix,
    df: pd.DataFrame,
    outputs_path: Path,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out_df = df.copy().reset_index(drop=True)
    out_df["predicted_dist"] = model.predict(X)
    apply_censor_cap(out_df, "censor", "predicted_dist", "score", "predicted_dist_censor_cap")

    csv_path = outputs_path / f"{split_name}_df_full.csv"
    out_df.to_csv(csv_path, index=False)

    metrics_path = outputs_path / f"{split_name}_prediction_metrics.txt"
    write_prediction_metrics(out_df, metrics_path)
    # Backward-compatible name used by some PLANT workflows.
    write_prediction_metrics(out_df, outputs_path / f"{split_name}_prediction_correlations.txt")

    metrics = compute_prediction_metrics_from_dataframe(out_df)
    print(
        f"{split_name} predicted_dist_censor_cap: "
        f"Pearson={metrics['pearson_censor_cap']:.4f}, "
        f"Spearman={metrics['spearman_censor_cap']:.4f}, "
        f"MAE={metrics['mae_censor_cap']:.4f}, "
        f"RMSE={metrics['rmse_censor_cap']:.4f}"
    )
    return out_df, metrics


def make_inner_cv_splits(
    X,
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create PLANT-matched inner CV splits for BayesSearchCV.

    BayesSearchCV normally calls ``cv.split(X, y, groups)``.  For a regression
    target, that would pass the continuous ``score`` values as ``y`` to
    StratifiedGroupKFold and fail with "Got 'continuous' instead".  Therefore
    we precompute the split indices using the same labels as PLANT:
    ``virus_collection_year`` for stratification and ``virus`` for grouping.
    """
    if n_splits < 2:
        raise ValueError(f"cv must be at least 2, got {n_splits}")

    groups = df["virus"].to_numpy()
    stratify_labels = df["virus_collection_year"].astype(int).to_numpy()

    unique_groups = pd.Series(groups).nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Inner CV n_splits={n_splits} is larger than the number of unique virus groups "
            f"({unique_groups}). Reduce --cv."
        )

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, stratify_labels, groups))


def make_bayes_search(seed: int, n_iter: int, cv, n_jobs: int, verbose: int) -> BayesSearchCV:
    ada = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(random_state=seed),
        random_state=seed,
        loss="linear",
    )

    search_space = {
        "n_estimators": Integer(150, 600),
        "learning_rate": Real(0.0005, 0.005, prior="log-uniform"),
        "estimator__max_depth": Integer(100, 750),
        "estimator__max_features": Categorical(["sqrt"]),
        "estimator__min_samples_leaf": Integer(1, 5),
    }

    return BayesSearchCV(
        estimator=ada,
        search_spaces=search_space,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=seed,
        refit=True,
        return_train_score=True,
    )


def clone_best_model(best_params: dict, seed: int) -> AdaBoostRegressor:
    tree_params = {"random_state": seed}
    ada_params = {"random_state": seed, "loss": "linear"}
    for key, value in best_params.items():
        if key.startswith("estimator__"):
            tree_params[key.replace("estimator__", "")] = value
        else:
            ada_params[key] = value
    return AdaBoostRegressor(estimator=DecisionTreeRegressor(**tree_params), **ada_params)


def summarize_split(
    *,
    args: argparse.Namespace,
    selected_df_with_season: pd.DataFrame,
    selected_df_past: pd.DataFrame,
    selected_df_future: pd.DataFrame | None,
    selected_df_train: pd.DataFrame,
    selected_df_val: pd.DataFrame,
    selected_df_test: pd.DataFrame,
    has_future: bool,
) -> dict:
    split_summary = {
        "split_mode": args.split_mode,
        "cutoff_season": parse_season_label(args.cutoff_season)[0] if args.split_mode == "season" else None,
        "season_col": args.season_col,
        "season_order_convention": "... SH2013 < NH2014 < SH2014 < NH2015 ...",
        "all_after_cleaning": len(selected_df_with_season),
        "past_total": len(selected_df_past),
        "future": len(selected_df_future) if has_future else 0,
        "train": len(selected_df_train),
        "validation": len(selected_df_val),
        "test": len(selected_df_test),
        "past_min_season": season_order_to_label(selected_df_past["season_order"].min()),
        "past_max_season": season_order_to_label(selected_df_past["season_order"].max()),
        "past_seasons": sorted(
            selected_df_past["season_normalized"].unique().tolist(),
            key=lambda s: parse_season_label(s)[1],
        ),
    }
    if has_future and selected_df_future is not None:
        split_summary.update(
            {
                "future_min_season": season_order_to_label(selected_df_future["season_order"].min()),
                "future_max_season": season_order_to_label(selected_df_future["season_order"].max()),
                "future_seasons": sorted(
                    selected_df_future["season_normalized"].unique().tolist(),
                    key=lambda s: parse_season_label(s)[1],
                ),
            }
        )
    else:
        split_summary.update(
            {
                "future_min_season": None,
                "future_max_season": None,
                "future_seasons": [],
            }
        )
    return split_summary


def determine_cutoff_seasons(args: argparse.Namespace, cleaned_df: pd.DataFrame) -> list[str | None]:
    if args.split_mode == "full":
        return [None]
    if args.all_cutoff_seasons:
        with_season = add_season_order_columns(cleaned_df, season_col="season")
        season_rows = (
            with_season[["season_normalized", "season_order"]]
            .drop_duplicates()
            .sort_values("season_order")
        )
        # The final season cannot be a cutoff in season mode because it would create no future set.
        return season_rows["season_normalized"].iloc[:-1].tolist()
    if args.cutoff_seasons:
        return [item.strip() for item in args.cutoff_seasons.split(",") if item.strip()]
    return [args.cutoff_season]


def save_feature_importance(
    model: AdaBoostRegressor,
    outputs_path: Path,
    metadata_encoders: dict[str, OneHotEncoder],
    *,
    max_length: int,
) -> None:
    importances = np.asarray(model.feature_importances_, dtype=float)
    site_importance = importances[:max_length]
    site_df = pd.DataFrame(
        {"site": np.arange(1, max_length + 1), "importance": site_importance}
    ).sort_values("importance", ascending=False)
    site_df.to_csv(outputs_path / "aaindex_site_feature_importance.csv", index=False)
    site_df.head(20).to_csv(outputs_path / "top20_aaindex_sites.csv", index=False)

    rows = []
    offset = max_length
    for col, encoder in metadata_encoders.items():
        cats = [str(x) for x in encoder.categories_[0]]
        for cat, imp in zip(cats, importances[offset : offset + len(cats)]):
            rows.append({"feature_group": col, "category": cat, "importance": float(imp)})
        offset += len(cats)
    if rows:
        pd.DataFrame(rows).sort_values("importance", ascending=False).to_csv(
            outputs_path / "metadata_feature_importance.csv", index=False
        )


def run_one_split(
    args: argparse.Namespace,
    *,
    cutoff_season: str | None,
    selected_df: pd.DataFrame,
    aaindex_matrix: np.ndarray,
) -> dict:
    if cutoff_season is not None:
        args.cutoff_season = cutoff_season

    storage_path = Path(args.directory).resolve()
    if args.split_mode == "season":
        output_split_label = f"trained_until_{parse_season_label(args.cutoff_season)[0]}"
    else:
        output_split_label = "full_model"
    outputs_path = storage_path / "Season_based_split_performance" / args.prefix / output_split_label
    outputs_path.mkdir(parents=True, exist_ok=True)

    selected_df_with_season = add_season_order_columns(selected_df, season_col="season")
    if args.split_mode == "season":
        selected_df_past, selected_df_future = split_past_future_by_cutoff_season(
            selected_df,
            cutoff_season=args.cutoff_season,
            season_col="season",
        )
        has_future = True
    else:
        selected_df_past = selected_df_with_season.reset_index(drop=True)
        selected_df_future = None
        has_future = False

    selected_df_train_val, selected_df_test = split_dataframe_with_stratified_group_kfold(
        selected_df_past, train_ratio=0.8, seed=args.random_seed
    )
    selected_df_test = selected_df_test.copy()
    selected_df_test["weight"] = 1.0
    if has_future and selected_df_future is not None:
        selected_df_future = selected_df_future.copy()
        selected_df_future["weight"] = 1.0

    if args.use_density_weights:
        selected_df_train_val = add_density_weights(selected_df_train_val)
    else:
        selected_df_train_val = selected_df_train_val.copy()
        selected_df_train_val["weight"] = 1.0

    selected_df_train, selected_df_val = split_dataframe_with_stratified_group_kfold(
        selected_df_train_val, train_ratio=8 / 9, seed=args.random_seed
    )

    split_summary = summarize_split(
        args=args,
        selected_df_with_season=selected_df_with_season,
        selected_df_past=selected_df_past,
        selected_df_future=selected_df_future,
        selected_df_train=selected_df_train,
        selected_df_val=selected_df_val,
        selected_df_test=selected_df_test,
        has_future=has_future,
    )
    split_summary.update(
        {
            "model_family": "AAindex_AdaBoostRegressor",
            "metadata_features": args.metadata_features,
            "tune_on": args.tune_on,
            "final_fit_on": args.final_fit_on,
            "use_density_weights": args.use_density_weights,
            "bayes_n_iter": args.n_iter,
            "bayes_cv": args.cv,
            "score_transform": args.score_transform,
        }
    )
    print(json.dumps(split_summary, indent=2))
    (outputs_path / "split_summary.json").write_text(json.dumps(split_summary, indent=2, sort_keys=True))

    # Save raw split membership for exact auditability.
    selected_df_train.to_csv(outputs_path / "train_input_df.csv", index=False)
    selected_df_val.to_csv(outputs_path / "validation_input_df.csv", index=False)
    selected_df_test.to_csv(outputs_path / "test_input_df.csv", index=False)
    if has_future and selected_df_future is not None:
        selected_df_future.to_csv(outputs_path / "future_input_df.csv", index=False)

    tune_df = selected_df_train if args.tune_on == "train" else selected_df_train_val
    if args.final_fit_on == "same_as_tune":
        final_fit_mode = args.tune_on
    else:
        final_fit_mode = args.final_fit_on
    final_fit_df = selected_df_train if final_fit_mode == "train" else selected_df_train_val

    metadata_cols = metadata_columns_for_mode(args.metadata_features)
    metadata_encoders = fit_metadata_encoders(tune_df, metadata_cols)

    print("Building feature matrices...")
    X_tune = build_feature_matrix(
        tune_df,
        aaindex_matrix=aaindex_matrix,
        metadata_encoders=metadata_encoders,
        max_length=args.max_length,
    )
    y_tune = tune_df["score"].astype(float).to_numpy()
    w_tune = tune_df["weight"].astype(float).to_numpy() if args.use_density_weights else None

    cv_splits = make_inner_cv_splits(X_tune, tune_df, args.cv, args.random_seed)
    bayes_search = make_bayes_search(args.random_seed, args.n_iter, cv_splits, args.n_jobs, args.verbose)

    print("Beginning BayesSearchCV...")
    fit_kwargs = {}
    if w_tune is not None:
        fit_kwargs["sample_weight"] = w_tune
    bayes_search.fit(X_tune, y_tune, **fit_kwargs)

    best_params = dict(bayes_search.best_params_)
    best_summary = {
        "best_params": {k: _json_safe_value(v) for k, v in best_params.items()},
        "best_score_neg_mae": _json_safe_value(bayes_search.best_score_),
        "best_cv_mae": _json_safe_value(-bayes_search.best_score_),
    }
    print(json.dumps(best_summary, indent=2, sort_keys=True))
    (outputs_path / "best_params.json").write_text(json.dumps(best_summary, indent=2, sort_keys=True))
    pd.DataFrame(bayes_search.cv_results_).to_csv(outputs_path / "bayes_cv_results.csv", index=False)
    with open(outputs_path / "bayes_search.pkl", "wb") as handle:
        pickle.dump(bayes_search, handle)

    # Refit the selected model on the requested final fit set. Refit may use train_val if explicitly requested.
    print(f"Refitting best model on {final_fit_mode}...")
    final_model = clone_best_model(best_params, args.random_seed)
    # Refit metadata encoders on the final fit data, because this is the actual deployable training input.
    metadata_encoders = fit_metadata_encoders(final_fit_df, metadata_cols)
    X_final_fit = build_feature_matrix(
        final_fit_df,
        aaindex_matrix=aaindex_matrix,
        metadata_encoders=metadata_encoders,
        max_length=args.max_length,
    )
    y_final_fit = final_fit_df["score"].astype(float).to_numpy()
    w_final_fit = final_fit_df["weight"].astype(float).to_numpy() if args.use_density_weights else None
    if w_final_fit is not None:
        final_model.fit(X_final_fit, y_final_fit, sample_weight=w_final_fit)
    else:
        final_model.fit(X_final_fit, y_final_fit)

    with open(outputs_path / "adaboost_model.pkl", "wb") as handle:
        pickle.dump(final_model, handle)
    for col, encoder in metadata_encoders.items():
        safe_col = col.replace("/", "_")
        joblib.dump(encoder, outputs_path / f"{safe_col}_encoder.joblib")
    (outputs_path / "feature_config.json").write_text(
        json.dumps(
            {
                "aa_order": AA_ORDER,
                "max_length": args.max_length,
                "metadata_columns": metadata_cols,
                "n_sequence_features": args.max_length,
                "n_metadata_features": int(
                    sum(len(encoder.categories_[0]) for encoder in metadata_encoders.values())
                ),
                "n_total_features": int(X_final_fit.shape[1]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    save_feature_importance(final_model, outputs_path, metadata_encoders, max_length=args.max_length)

    split_metrics = {}
    for split_name, split_df in [
        ("train", selected_df_train),
        ("validation", selected_df_val),
        ("test", selected_df_test),
    ]:
        X_split = build_feature_matrix(
            split_df,
            aaindex_matrix=aaindex_matrix,
            metadata_encoders=metadata_encoders,
            max_length=args.max_length,
        )
        _, metrics = predict_and_save_split(final_model, X_split, split_df, outputs_path, split_name=split_name)
        split_metrics[split_name] = {k: _json_safe_value(v) for k, v in metrics.items()}

    if has_future and selected_df_future is not None:
        X_future = build_feature_matrix(
            selected_df_future,
            aaindex_matrix=aaindex_matrix,
            metadata_encoders=metadata_encoders,
            max_length=args.max_length,
        )
        future_df, metrics = predict_and_save_split(final_model, X_future, selected_df_future, outputs_path, split_name="future")
        split_metrics["future"] = {k: _json_safe_value(v) for k, v in metrics.items()}
        write_future_metrics_by_season(future_df, outputs_path, season_col="season_normalized")

    (outputs_path / "all_split_metrics.json").write_text(json.dumps(split_metrics, indent=2, sort_keys=True))
    return {
        "output": str(outputs_path),
        "best_cv_mae": float(-bayes_search.best_score_),
        "test_mae_censor_cap": split_metrics.get("test", {}).get("mae_censor_cap"),
        "test_pearson_censor_cap": split_metrics.get("test", {}).get("pearson_censor_cap"),
        "future_mae_censor_cap": split_metrics.get("future", {}).get("mae_censor_cap"),
        "future_pearson_censor_cap": split_metrics.get("future", {}).get("pearson_censor_cap"),
    }


def main() -> None:
    start = time.time()
    args = parse_args()
    set_seed(args.random_seed)

    storage_path = Path(args.directory).resolve()
    paired_csv = (
        Path(args.paired_csv)
        if args.paired_csv
        else storage_path / "data" / "WHO_GISAID_dataset_final_strict_score_250124.csv"
    )
    aaindex_csv = resolve_aaindex_path(storage_path, args.aaindex_csv)

    print(f"Paired CSV: {paired_csv}")
    print(f"AAindex CSV: {aaindex_csv}")
    print("Loading and cleaning paired data once...")
    selected_df = clean_paired_dataset(
        paired_csv,
        args.max_length,
        args.random_seed,
        season_col=args.season_col,
        score_transform=args.score_transform,
    )
    aaindex_matrix = load_aaindex(aaindex_csv)

    cutoff_seasons = determine_cutoff_seasons(args, selected_df)
    print(f"Cutoff seasons to run: {cutoff_seasons}")

    run_summaries = []
    for cutoff in cutoff_seasons:
        print("=" * 80)
        print(f"Running split: {cutoff if cutoff is not None else 'full_model'}")
        run_summaries.append(
            run_one_split(
                args,
                cutoff_season=cutoff,
                selected_df=selected_df,
                aaindex_matrix=aaindex_matrix,
            )
        )

    summary_df = pd.DataFrame(run_summaries)
    summary_dir = storage_path / "Season_based_split_performance" / args.prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_dir / "adaboost_time_split_summary.csv", index=False)
    print(summary_df)
    print(f"Done in {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    main()

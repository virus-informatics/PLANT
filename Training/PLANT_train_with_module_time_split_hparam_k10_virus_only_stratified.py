#!/usr/bin/env python
"""Train PLANT using the reusable ``plant`` Python module.

Run from the repository root, for example:

python Training/PLANT_train_with_module.py \
  --prefix full_module_train \
  --directory . \
  --batch-size 16 \
  --num-steps 20000 \
  --model facebook/esm2_t36_3B_UR50D
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer, EsmConfig, TrainingArguments

# Make src/plant importable when this script is run from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from plant import (  # noqa: E402
    BalancedCombinationTrainer,
    TextDataset,
    build_plant_optimizer,
    compute_embedding_distances,
    embed_sequences,
    estimate_embed_scale_factor,
    semanticESM,
    set_encoders,
    tokenize_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="PLANT trainer",
        description="Train PLANT with reusable module components.",
    )
    parser.add_argument("--prefix", required=True, help="Output prefix/version name.")
    parser.add_argument(
        "--directory",
        default=".",
        help="Repository or project directory containing the data/ folder.",
    )
    parser.add_argument(
        "--paired-csv",
        default=None,
        help="CSV for paired antigenic-distance training data. Default: data/WHO_GISAID_dataset_final_strict_score_250124.csv",
    )
    parser.add_argument(
        "--sequence-pool-fasta",
        default=None,
        help="FASTA for virus-only semantic training. Default: data/ncbiflu_HA_all_110424_noX_clu99_aln_realign2.fas",
    )
    parser.add_argument(
        "--sequence-pool-metadata",
        default=None,
        help="Optional metadata CSV for the sequence-pool FASTA. Default: data/ncbiflu_HA_all_110424_noX_clu99_meta.csv",
    )
    parser.add_argument(
        "--gisaid-csv",
        default=None,
        help=(
            "CSV whose seq column will be embedded after training. "
            "Default: data/PLANT_epiflu_human_241212.csv, matching the original trainer."
        ),
    )
    parser.add_argument(
        "--skip-gisaid",
        action="store_true",
        help="Skip embedding the default/all-sequence GISAID CSV after training.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--model", default="facebook/esm2_t36_3B_UR50D")
    parser.add_argument("--max-length", default=329, type=int)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", default=16, type=int)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", default=32, type=int)
    parser.add_argument("--embedding-batch-size", "--embedding_batch_size", dest="embedding_batch_size", default=128, type=int)
    parser.add_argument("--num-steps", "--num_steps", dest="num_steps", default=20000, type=int)
    parser.add_argument("--random-seed", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", default=0.01, type=float)
    parser.add_argument("--reg-weight-decay", "--reg_weight_decay", dest="reg_weight_decay", default=0.01, type=float)
    parser.add_argument("--max-saves", "--max_saves", dest="max_saves", default=1, type=int)
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", default=1000, type=int)
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", default=1000, type=int)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num-samples-per-combination", "--num_samples_per_combination", dest="num_samples_per_combination", default=1, type=int)
    parser.add_argument("--CSE-w", "--CSE_w", dest="CSE_w", default=0.0, type=float)
    parser.add_argument("--CSE-w-virus-only", "--CSE_w_virus_only", dest="CSE_w_virus_only", default=0.0, type=float)
    parser.add_argument("--semantic-w", "--semantic_w", dest="semantic_w", default=0.2, type=float)
    parser.add_argument("--semantic-w-virus-only", "--semantic_w_virus_only", dest="semantic_w_virus_only", default=0.2, type=float)
    parser.add_argument("--cart-w", "--cart_w", dest="cart_w", default=0.05, type=float)
    parser.add_argument("--dropout-regressor", "--dropout_regressor", dest="dropout_regressor", default=0.05, type=float)
    parser.add_argument("--reg-intermediate-dim", "--reg_intermediate_dim", dest="reg_intermediate_dim", default=256, type=int)
    parser.add_argument("--CSE-alpha", "--CSE_alpha", dest="CSE_alpha", default=0.0, type=float)
    parser.add_argument("--intermediate-dim-encoder", "--intermediate_dim_encoder", dest="intermediate_dim_encoder", default=64, type=int)
    parser.add_argument("--dropout-encoder", "--dropout_encoder", dest="dropout_encoder", default=0.1, type=float)
    parser.add_argument("--lg-w", "--lg_w", dest="lg_w", default=0.0, type=float)
    parser.add_argument(
        "--reference-transform-mode",
        "--reference_transform_mode",
        dest="reference_transform_mode",
        choices=["none", "full", "diagonal"],
        default="full",
        help=(
            "Reference-side coordinate transform. 'full' learns a near-identity "
            "affine transform, 'diagonal' learns axis-wise scaling plus shift, "
            "and 'none' restores the shared-coordinate baseline."
        ),
    )
    parser.add_argument(
        "--ref-transform-w",
        "--ref_transform_w",
        dest="ref_transform_w",
        default=0.05,
        type=float,
        help="Weight for keeping the reference transform close to identity.",
    )
    parser.add_argument(
        "--ref-shift-w",
        "--ref_shift_w",
        dest="ref_shift_w",
        default=0.05,
        type=float,
        help="Weight for penalizing the data-scale shift of transformed reference points.",
    )
    parser.add_argument(
        "--time-cutoff-date",
        "--time_cutoff_date",
        dest="time_cutoff_date",
        default="2022-08-31",
        help=(
            "Cutoff date for past/future evaluation split based on the paired-data "
            "'date' column. Rows with resolved dates <= cutoff are used for the "
            "random train/validation/test split; rows after cutoff are held out "
            "as the future dataset. Incomplete dates are resolved conservatively "
            "to the latest possible date."
        ),
    )
    parser.add_argument(
        "--virus-only-ratio-k",
        "--virus_only_ratio_k",
        dest="virus_only_ratio_k",
        default=10,
        type=int,
        help=(
            "Randomly sample virus-only sequences so that their count is "
            "approximately 1/k of the paired training dataset. "
            "Use 0 or a negative value to disable subsampling."
        ),
    )
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fasta_to_dataframe(fasta_file: Path) -> pd.DataFrame:
    """Read FASTA without requiring Biopython."""
    ids = []
    sequences = []
    current_id = None
    current_seq = []
    with fasta_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    sequences.append("".join(current_seq))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        ids.append(current_id)
        sequences.append("".join(current_seq))
    return pd.DataFrame({"ID": ids, "seq": sequences})


def complete_date(date_str) -> str:
    if pd.isna(date_str):
        return date_str
    date_str = str(date_str)
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    parts = date_str.split("-")
    try:
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit() and date_str.endswith("-"):
            year, month = int(parts[0]), int(parts[1])
            last_day = calendar.monthrange(year, month)[1]
            return f"{year:04d}-{month:02d}-{last_day:02d}"
        if date_str.count("-") == 2 and parts[0].isdigit() and parts[1] == "" and parts[2] == "":
            return f"{int(parts[0]):04d}-12-31"
        if len(parts) == 1 and parts[0].isdigit():
            return f"{int(parts[0]):04d}-12-31"
    except ValueError:
        pass
    return date_str



def resolve_partial_date_for_time_split(date_value) -> pd.Timestamp:
    """Resolve incomplete date strings to the latest plausible date.

    This is used only for the past/future split.  The paired dataset can contain
    partially specified or non-standard strings such as ``1993-aaa``.  To avoid
    leaking ambiguous records into the past set, missing/invalid month and day
    components are filled with the latest possible values: month=12 and
    day=last day of month.  Examples:

    - ``1993-aaa`` -> ``1993-12-31``
    - ``2024-01-aaa`` -> ``2024-01-31``
    - ``2024-01`` -> ``2024-01-31``
    """
    if pd.isna(date_value):
        return pd.NaT

    raw = str(date_value).strip()
    if not raw:
        return pd.NaT

    parts = raw.split("-")
    if not parts or not parts[0].isdigit():
        return pd.NaT

    try:
        year = int(parts[0])
    except ValueError:
        return pd.NaT

    if year < 1:
        return pd.NaT

    month = 12
    if len(parts) >= 2 and parts[1].isdigit():
        parsed_month = int(parts[1])
        if 1 <= parsed_month <= 12:
            month = parsed_month

    last_day = calendar.monthrange(year, month)[1]
    day = last_day
    if len(parts) >= 3 and parts[2].isdigit():
        parsed_day = int(parts[2])
        if 1 <= parsed_day <= last_day:
            day = parsed_day

    return pd.Timestamp(year=year, month=month, day=day)


def split_past_future_by_cutoff(
    df: pd.DataFrame,
    *,
    cutoff_date: str,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split paired data into past and future sets using a cutoff date.

    Rows with resolved dates <= cutoff are assigned to the past set.  Rows with
    resolved dates > cutoff are assigned to the future set.
    """
    cutoff = pd.Timestamp(cutoff_date)
    out = df.copy()
    out["date_for_time_split"] = out[date_col].apply(resolve_partial_date_for_time_split)

    unparseable = out["date_for_time_split"].isna()
    if unparseable.any():
        print(
            "Dropping rows with unparseable date values for time split: "
            f"{int(unparseable.sum())}"
        )
        out = out.loc[~unparseable].copy()

    past = out.loc[out["date_for_time_split"] <= cutoff].copy()
    future = out.loc[out["date_for_time_split"] > cutoff].copy()

    if past.empty:
        raise ValueError(
            f"No rows were assigned to the past set using cutoff_date={cutoff_date!r}."
        )
    if future.empty:
        raise ValueError(
            f"No rows were assigned to the future set using cutoff_date={cutoff_date!r}."
        )

    return past.reset_index(drop=True), future.reset_index(drop=True)


def load_sequence_pool(fasta_path: Path, metadata_path: Path | None) -> pd.DataFrame:
    df = fasta_to_dataframe(fasta_path)
    df["seq"] = df["seq"].str.replace("-", "", regex=False)
    if metadata_path is not None and metadata_path.exists():
        metadata = pd.read_csv(metadata_path).rename(columns={"name": "ID"})
        df = pd.merge(df, metadata, on="ID", how="inner")
        if "date" in df.columns:
            df["year"] = df["date"].astype(str).str.slice(0, 4)
            df = df[df["year"].str.isnumeric()].copy()
            df["year"] = df["year"].astype(int)
            df["date_corrected"] = pd.to_datetime(
                df["date"].apply(complete_date), errors="coerce"
            )
        if "serotype" in df.columns:
            df["HA_type"] = df["serotype"].astype(str).str.slice(0, 2)
    df = df[~df["seq"].str.contains("X", regex=False, na=False)].copy()
    return df.reset_index(drop=True)


def _allocate_stratified_sample_counts(group_sizes: pd.Series, sample_n: int) -> pd.Series:
    """Allocate sample counts across strata using largest-remainder rounding."""
    if sample_n <= 0:
        raise ValueError(f"sample_n must be positive: {sample_n}")
    if group_sizes.empty:
        raise ValueError("group_sizes is empty.")

    ideal = group_sizes.astype(float) / float(group_sizes.sum()) * float(sample_n)
    allocated = np.floor(ideal).astype(int)
    remaining = int(sample_n - allocated.sum())

    # Distribute remaining slots to strata with the largest fractional remainders,
    # while respecting each stratum's available count.
    while remaining > 0:
        capacity = group_sizes - allocated
        candidates = capacity[capacity > 0].index
        if len(candidates) == 0:
            break
        order = (ideal.loc[candidates] - allocated.loc[candidates]).sort_values(ascending=False).index
        for key in order:
            if remaining <= 0:
                break
            if allocated.loc[key] < group_sizes.loc[key]:
                allocated.loc[key] += 1
                remaining -= 1

    return allocated[allocated > 0]


def sample_virus_only_pool(
    sequence_pool_df: pd.DataFrame,
    *,
    train_size: int,
    k: int,
    seed: int,
    stratify_cols: tuple[str, ...] = ("host", "serotype"),
) -> pd.DataFrame:
    """Randomly subsample virus-only sequences to approximately train_size / k.

    If all columns in ``stratify_cols`` are present, sampling is stratified by their
    combinations so that the sampled pool approximately preserves the original
    host/serotype composition. If any required column is missing, the function
    falls back to simple random sampling.
    """
    sequence_pool_df = sequence_pool_df.reset_index(drop=True)
    available_n = len(sequence_pool_df)
    if available_n == 0:
        raise ValueError("Virus-only sequence pool is empty after filtering.")

    if k <= 0:
        print(
            "Virus-only subsampling disabled: "
            f"using all {available_n} available sequences."
        )
        return sequence_pool_df

    if train_size <= 0:
        raise ValueError(f"Training dataset is empty or invalid: train_size={train_size}")

    target_n = max(1, int(train_size / k + 0.5))
    sample_n = min(target_n, available_n)
    if sample_n < target_n:
        print(
            "Virus-only sequence pool is smaller than requested: "
            f"target={target_n}, available={available_n}. Using all available sequences."
        )

    print(
        "Sampling virus-only sequence pool: "
        f"train_size={train_size}, k={k}, target={target_n}, sampled={sample_n}, "
        f"available={available_n}"
    )

    missing_cols = [col for col in stratify_cols if col not in sequence_pool_df.columns]
    if missing_cols:
        print(
            "Stratified sampling disabled because required columns are missing: "
            f"{missing_cols}. Falling back to simple random sampling."
        )
        return sequence_pool_df.sample(n=sample_n, random_state=seed, replace=False).reset_index(drop=True)

    stratified_df = sequence_pool_df.copy()
    strata_cols = []
    for col in stratify_cols:
        strata_col = f"__strata_{col}"
        strata_cols.append(strata_col)
        stratified_df[strata_col] = stratified_df[col].astype("object").where(
            stratified_df[col].notna(), "__MISSING__"
        )

    group_sizes = stratified_df.groupby(strata_cols, dropna=False).size()
    sample_counts = _allocate_stratified_sample_counts(group_sizes, sample_n)

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for key, n in sample_counts.items():
        if not isinstance(key, tuple):
            key = (key,)
        mask = np.ones(len(stratified_df), dtype=bool)
        for col, value in zip(strata_cols, key):
            mask &= stratified_df[col].to_numpy() == value
        group_df = stratified_df.loc[mask]
        sampled_parts.append(
            group_df.sample(n=int(n), random_state=int(rng.integers(0, 2**32 - 1)), replace=False)
        )

    sampled_df = pd.concat(sampled_parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled_df = sampled_df.drop(columns=strata_cols)
    print(
        "Stratified virus-only sampling by "
        f"{list(stratify_cols)}: strata={len(group_sizes)}, sampled_strata={len(sample_counts)}"
    )
    return sampled_df


def clean_paired_dataset(path: Path, max_length: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = [
        "date",
        "virus",
        "reference",
        "virus_passage",
        "reference_passage",
        "score",
        "censor",
        "virus_seq",
        "reference_seq",
        "virus_collection_date",
        "reference_collection_date",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for col in ["virus_seq", "reference_seq"]:
        df = df[~df[col].str.contains("X", regex=False, na=False)]
        df = df[~df[col].str.contains("B", regex=False, na=False)]
        df = df[~df[col].str.contains("*", regex=False, na=False)]
    if "pair_strict" in df.columns:
        df = df[~df["pair_strict"].str.contains("|", regex=False, na=False)]

    df = df[df["virus_seq"].str.len() == max_length]
    df = df[df["reference_seq"].str.len() == max_length]

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


def make_paired_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> TextDataset:
    encodes_virus = tokenize_sequences(df["virus_seq"].tolist(), tokenizer, max_length)
    encodes_reference = tokenize_sequences(df["reference_seq"].tolist(), tokenizer, max_length)
    return TextDataset(
        encodes_virus,
        encodes_reference,
        labels=df["score"].astype(float).tolist(),
        censors=df["censor"].astype(float).tolist(),
        virus=df["virus_category"].astype(int).tolist(),
        reference=df["reference_category"].astype(int).tolist(),
        dates=df["date_category"].astype(int).tolist(),
        virus_passage=df["virus_passage_category"].astype(int).tolist(),
        reference_passage=df["reference_passage_category"].astype(int).tolist(),
        weight=df["weight"].astype(float).tolist(),
    )


def make_virus_only_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> TextDataset:
    encodes = tokenize_sequences(df["seq"].tolist(), tokenizer, max_length)
    # Provide one group per sequence ID so balanced sampling uses the unlabeled pool.
    virus_ids = list(range(len(df)))
    return TextDataset(encodes, virus=virus_ids)


def make_training_args(**kwargs) -> TrainingArguments:
    """Handle the evaluation_strategy -> eval_strategy rename across transformers versions."""
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:
        if "eval_strategy" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
        return TrainingArguments(**kwargs)


def apply_censor_cap(df: pd.DataFrame, censor_col: str, predicted_col: str, score_col: str, output_col: str) -> None:
    df[output_col] = np.where(
        (df[censor_col].astype(float) == 1) & (df[predicted_col] > df[score_col]),
        df[score_col],
        df[predicted_col],
    )


PREDICTION_COLUMNS = [
    "predicted_dist",
    "predicted_dist_censor_cap",
    "predicted_dist_cartography",
    "predicted_dist_cartography_censor_cap",
]


def _safe_correlation(x, y, *, method: str) -> tuple[float, float, int]:
    """Return correlation, p-value, and N while tolerating NaNs/constant vectors."""
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
    """Return MAE/RMSE and N while tolerating NaNs."""
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
    """Fraction of censored examples whose prediction violates the lower bound."""
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


def _extract_logits(predictions) -> np.ndarray:
    """Extract the (observed distance, cartography distance) logits from Trainer outputs."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    logits = np.asarray(predictions)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.shape[1] < 2:
        raise ValueError(
            "Expected model predictions with at least two columns: "
            "observed distance and cartography distance."
        )
    return logits[:, :2]


def _extract_labels_and_censors(label_ids) -> tuple[np.ndarray, np.ndarray]:
    """Extract labels and censors from EvalPrediction.label_ids.

    When TrainingArguments(label_names=["labels", "censors"]) is used, Hugging Face
    passes a tuple/list of arrays.  This helper also supports the older fallback
    where only labels are available, in which case censors are treated as zeros.
    """
    if isinstance(label_ids, (tuple, list)):
        labels = np.asarray(label_ids[0], dtype=float).reshape(-1)
        if len(label_ids) > 1:
            censors = np.asarray(label_ids[1], dtype=float).reshape(-1)
        else:
            censors = np.zeros_like(labels, dtype=float)
    else:
        labels = np.asarray(label_ids, dtype=float).reshape(-1)
        censors = np.zeros_like(labels, dtype=float)
    return labels, censors


def add_prediction_columns_from_logits(
    df: pd.DataFrame,
    logits: np.ndarray,
    *,
    score_col: str = "score",
    censor_col: str = "censor",
) -> pd.DataFrame:
    """Add PLANT prediction columns from model logits to a dataframe."""
    out = df.copy().reset_index(drop=True)
    logits = _extract_logits(logits)
    if len(out) != logits.shape[0]:
        raise ValueError(f"Dataframe length ({len(out)}) != logits rows ({logits.shape[0]}).")

    out["predicted_dist"] = logits[:, 0]
    out["predicted_dist_cartography"] = logits[:, 1]
    apply_censor_cap(out, censor_col, "predicted_dist", score_col, "predicted_dist_censor_cap")
    apply_censor_cap(
        out,
        censor_col,
        "predicted_dist_cartography",
        score_col,
        "predicted_dist_cartography_censor_cap",
    )
    return out


def compute_prediction_metrics(
    score,
    censor,
    prediction_columns: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute correlation and error metrics for PLANT prediction columns."""
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

    # Short aliases for checkpoint selection and compact logging.
    if "predicted_dist_censor_cap" in prediction_columns:
        metrics["pearson_censor_cap"] = metrics["pearson_predicted_dist_censor_cap"]
        metrics["spearman_censor_cap"] = metrics["spearman_predicted_dist_censor_cap"]
        metrics["mae_censor_cap"] = metrics["mae_predicted_dist_censor_cap"]
        metrics["rmse_censor_cap"] = metrics["rmse_predicted_dist_censor_cap"]

    return metrics


def compute_metrics_for_trainer(eval_pred) -> dict[str, float]:
    """Compute validation metrics used by Hugging Face Trainer."""
    logits = _extract_logits(eval_pred.predictions)
    labels, censors = _extract_labels_and_censors(eval_pred.label_ids)

    observed_distance = logits[:, 0]
    cartography_distance = logits[:, 1]
    predicted_dist_censor_cap = np.where(
        (censors == 1) & (observed_distance > labels),
        labels,
        observed_distance,
    )
    predicted_dist_cartography_censor_cap = np.where(
        (censors == 1) & (cartography_distance > labels),
        labels,
        cartography_distance,
    )

    return compute_prediction_metrics(
        labels,
        censors,
        {
            "predicted_dist": observed_distance,
            "predicted_dist_censor_cap": predicted_dist_censor_cap,
            "predicted_dist_cartography": cartography_distance,
            "predicted_dist_cartography_censor_cap": predicted_dist_cartography_censor_cap,
        },
    )


def compute_prediction_metrics_from_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """Compute prediction metrics from a dataframe containing score/censor/prediction columns."""
    available_predictions = {col: df[col].to_numpy() for col in PREDICTION_COLUMNS if col in df.columns}
    return compute_prediction_metrics(
        df["score"].to_numpy(),
        df["censor"].to_numpy(),
        available_predictions,
    )


def write_prediction_metrics(df: pd.DataFrame, output_path: Path) -> None:
    """Write Pearson/Spearman correlations and error metrics for prediction columns."""
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
                (
                    "censored_lower_bound_violation_rate\t"
                    f"{metrics[f'censored_lower_bound_violation_rate_{col}']:.8g}"
                ),
                f"censored_lower_bound_violation_n\t{int(metrics[f'censored_lower_bound_violation_n_{col}'])}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines))



def predict_and_save_split(
    trainer: BalancedCombinationTrainer,
    dataset: TextDataset,
    df: pd.DataFrame,
    outputs_path: Path,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predict one held-out split, save predictions/metrics, and return metrics."""
    print(f"Predicting {split_name} data...")
    predictions = trainer.predict(dataset)
    logits = _extract_logits(predictions.predictions)
    out_df = add_prediction_columns_from_logits(df, logits)

    if isinstance(predictions.predictions, tuple) and len(predictions.predictions) > 1:
        latent = predictions.predictions[1]
        out_df[["z1", "z2", "z3"]] = latent[:, :3]

    csv_path = outputs_path / f"{split_name}_df_full.csv"
    out_df.to_csv(csv_path, index=False)

    metrics_file = outputs_path / f"{split_name}_prediction_metrics.txt"
    write_prediction_metrics(out_df, metrics_file)

    # Backward-compatible filename for workflows that still look for the old output.
    legacy_corr_file = outputs_path / f"{split_name}_prediction_correlations.txt"
    write_prediction_metrics(out_df, legacy_corr_file)

    metrics = compute_prediction_metrics_from_dataframe(out_df)
    corr = metrics["pearson_censor_cap"]
    spearman = metrics["spearman_censor_cap"]
    mae = metrics["mae_censor_cap"]
    rmse = metrics["rmse_censor_cap"]
    print(
        f"{split_name} metrics for predicted_dist_censor_cap: "
        f"Pearson={corr:.4f}, Spearman={spearman:.4f}, "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}"
    )
    print(f"Saved {split_name} predictions: {csv_path}")
    print(f"Saved {split_name} metrics: {metrics_file}")
    print(f"Saved {split_name} legacy metrics/correlations: {legacy_corr_file}")

    return out_df, metrics


def main() -> None:
    start = time.time()
    args = parse_args()
    set_seed(args.random_seed)

    storage_path = Path(args.directory).resolve()
    outputs_path = storage_path / "Season_based_split_performance" / args.prefix / "trained_until_full"
    outputs_path.mkdir(parents=True, exist_ok=True)

    bf16 = (
        (not args.no_fp16)
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    fp16 = (
        (not args.no_fp16)
        and torch.cuda.is_available()
        and not bf16
    )

    paired_csv = Path(args.paired_csv) if args.paired_csv else storage_path / "data" / "WHO_GISAID_dataset_final_strict_score_250124.csv"
    sequence_pool_fasta = (
        Path(args.sequence_pool_fasta)
        if args.sequence_pool_fasta
        else storage_path / "data" / "ncbiflu_HA_all_110424_noX_clu99_aln_realign2.fas"
    )
    sequence_pool_metadata = (
        Path(args.sequence_pool_metadata)
        if args.sequence_pool_metadata
        else storage_path / "data" / "ncbiflu_HA_all_110424_noX_clu99_meta.csv"
    )

    print(f"Model: {args.model}")
    print(f"Output: {outputs_path}")
    print(f"Mixed precision: bf16={bf16}, fp16={fp16}")
    print(
        "Reference transform: "
        f"mode={args.reference_transform_mode}, "
        f"REF_TRANSFORM_W={args.ref_transform_w}, "
        f"REF_SHIFT_W={args.ref_shift_w}"
    )
    print("Loading paired antigenic-distance data...")
    selected_df = clean_paired_dataset(paired_csv, args.max_length, args.random_seed)
    selected_df_past, selected_df_future = split_past_future_by_cutoff(
        selected_df,
        cutoff_date=args.time_cutoff_date,
        date_col="date",
    )

    selected_df_train_val, selected_df_test = split_dataframe_with_stratified_group_kfold(
        selected_df_past, train_ratio=0.8, seed=args.random_seed
    )
    selected_df_test = selected_df_test.copy()
    selected_df_test["weight"] = 1.0
    selected_df_future = selected_df_future.copy()
    selected_df_future["weight"] = 1.0

    selected_df_train_val = add_density_weights(selected_df_train_val)
    selected_df_train, selected_df_val = split_dataframe_with_stratified_group_kfold(
        selected_df_train_val, train_ratio=8 / 9, seed=args.random_seed
    )

    split_summary = {
        "time_cutoff_date": args.time_cutoff_date,
        "all_after_cleaning": len(selected_df),
        "past_total": len(selected_df_past),
        "future": len(selected_df_future),
        "train": len(selected_df_train),
        "validation": len(selected_df_val),
        "test": len(selected_df_test),
        "past_min_date": str(selected_df_past["date_for_time_split"].min().date()),
        "past_max_date": str(selected_df_past["date_for_time_split"].max().date()),
        "future_min_date": str(selected_df_future["date_for_time_split"].min().date()),
        "future_max_date": str(selected_df_future["date_for_time_split"].max().date()),
    }
    print(json.dumps(split_summary, indent=2))
    (outputs_path / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True)
    )

    print("Loading virus-only sequence pool...")
    sequence_pool_df = load_sequence_pool(sequence_pool_fasta, sequence_pool_metadata)
    sequence_pool_df = sequence_pool_df.reset_index(drop=True)
    print(f"Virus-only sequence pool size before sampling: {len(sequence_pool_df)}")
    sequence_pool_df = sample_virus_only_pool(
        sequence_pool_df,
        train_size=len(selected_df_train),
        k=args.virus_only_ratio_k,
        seed=args.random_seed,
    )
    print(f"Virus-only sequence pool size after sampling: {len(sequence_pool_df)}")
    sequence_pool_df.to_csv(outputs_path / "virus_only_sequence_pool_sampled.csv", index=False)

    print("Tokenizing sequences...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset_train = make_paired_dataset(selected_df_train, tokenizer, args.max_length)
    dataset_val = make_paired_dataset(selected_df_val, tokenizer, args.max_length)
    dataset_test = make_paired_dataset(selected_df_test, tokenizer, args.max_length)
    dataset_future = make_paired_dataset(selected_df_future, tokenizer, args.max_length)
    dataset_virus_only = make_virus_only_dataset(sequence_pool_df, tokenizer, args.max_length)
    dataset_train_combined = ConcatDataset([dataset_train, dataset_virus_only])

    print("Fitting one-hot encoders...")
    ohe_virus = make_one_hot_encoder(selected_df_train["virus_category"])
    ohe_ref = make_one_hot_encoder(selected_df_train["reference_category"])
    ohe_date = make_one_hot_encoder(selected_df_train["date_category"])
    ohe_vp = make_one_hot_encoder(selected_df_train["virus_passage_category"])
    ohe_rp = make_one_hot_encoder(selected_df_train["reference_passage_category"])
    set_encoders(ohe_virus, ohe_ref, ohe_vp, ohe_rp)

    joblib.dump(ohe_virus, outputs_path / "virus_encoder.joblib")
    joblib.dump(ohe_ref, outputs_path / "ref_encoder.joblib")
    joblib.dump(ohe_date, outputs_path / "date_encoder.joblib")
    joblib.dump(ohe_vp, outputs_path / "vp_encoder.joblib")
    joblib.dump(ohe_rp, outputs_path / "rp_encoder.joblib")

    effects_len = len(ohe_ref.categories_[0]) + len(ohe_vp.categories_[0]) + len(ohe_rp.categories_[0])
    virus_effects_len = len(ohe_virus.categories_[0])

    print("Computing frozen ESM embedding distances...")
    embed_dist_train = compute_embedding_distances(
        dataset_train,
        esm_model_name=args.model,
        batch_size=args.embedding_batch_size,
        use_bf16=bf16,
        use_fp16=fp16,
    )
    selected_df_train = selected_df_train.copy()
    selected_df_train["embed_dist"] = embed_dist_train
    embed_scale_factor = float(np.quantile(embed_dist_train, 0.99))
    print(f"embed_scale_factor: {embed_scale_factor:.6g}")

    # The original trainer also records embed_dist for the held-out test table.
    embed_dist_test = compute_embedding_distances(
        dataset_test,
        esm_model_name=args.model,
        batch_size=args.embedding_batch_size,
        use_bf16=bf16,
        use_fp16=fp16,
    )
    selected_df_test = selected_df_test.copy()
    selected_df_test["embed_dist"] = embed_dist_test

    embed_dist_future = compute_embedding_distances(
        dataset_future,
        esm_model_name=args.model,
        batch_size=args.embedding_batch_size,
        use_bf16=bf16,
        use_fp16=fp16,
    )
    selected_df_future = selected_df_future.copy()
    selected_df_future["embed_dist"] = embed_dist_future

    print("Building PLANT model...")
    esm_config = EsmConfig.from_pretrained(args.model, use_safetensors=True)
    model = semanticESM(
        esm_config,
        args.model,
        effects_len=effects_len,
        virus_effects_len=virus_effects_len,
        embed_scale_factor=embed_scale_factor,
        CSE_W=args.CSE_w,
        CSE_W_VIRUS_ONLY=args.CSE_w_virus_only,
        SEMANTIC_W=args.semantic_w,
        SEMANTIC_W_VIRUS_ONLY=args.semantic_w_virus_only,
        MAIN_W=1.0,
        CART_W=args.cart_w,
        intermediate_dim=args.reg_intermediate_dim,
        dropout=args.dropout_regressor,
        dropout_encoder=args.dropout_encoder,
        intermediate_dim_encoder=args.intermediate_dim_encoder,
        CSE_ALPHA=args.CSE_alpha,
        LG_W=args.lg_w,
        reference_transform_mode=args.reference_transform_mode,
        REF_TRANSFORM_W=args.ref_transform_w,
        REF_SHIFT_W=args.ref_shift_w,
    )

    optimizer = build_plant_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regressor_weight_decay=args.reg_weight_decay,
    )

    training_args = make_training_args(
        output_dir=str(outputs_path / "results"),
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
        bf16=bf16,
        fp16=fp16,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.max_saves,
        warmup_ratio=args.warmup_ratio,
        logging_dir=str(outputs_path / "logs"),
        remove_unused_columns=False,
        label_names=["labels", "censors"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_pearson_censor_cap",
        greater_is_better=True,
        report_to="none",
    )

    trainer = BalancedCombinationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_combined,
        eval_dataset=dataset_val,
        num_samples_per_combination=args.num_samples_per_combination,
        random_seed=args.random_seed,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics_for_trainer,
    )

    print("Starting training...")
    if args.checkpoint:
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        trainer.train()
    print("Training completed.")

    print("Saving final model and tokenizer...")
    final_model_dir = outputs_path / "model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    run_config = vars(args).copy()
    run_config.update(
        {
            "bf16": bf16,
            "fp16": fp16,
            "embed_scale_factor": embed_scale_factor,
            "model_dir": str(final_model_dir),
            "split_summary": split_summary,
        }
    )
    (outputs_path / "training_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True)
    )

    selected_df_test, test_metrics = predict_and_save_split(
        trainer,
        dataset_test,
        selected_df_test,
        outputs_path,
        split_name="test",
    )
    selected_df_future, future_metrics = predict_and_save_split(
        trainer,
        dataset_future,
        selected_df_future,
        outputs_path,
        split_name="future",
    )
    heldout_summary = {
        "test": {
            "pearson_censor_cap": test_metrics["pearson_censor_cap"],
            "spearman_censor_cap": test_metrics["spearman_censor_cap"],
            "mae_censor_cap": test_metrics["mae_censor_cap"],
            "rmse_censor_cap": test_metrics["rmse_censor_cap"],
        },
        "future": {
            "pearson_censor_cap": future_metrics["pearson_censor_cap"],
            "spearman_censor_cap": future_metrics["spearman_censor_cap"],
            "mae_censor_cap": future_metrics["mae_censor_cap"],
            "rmse_censor_cap": future_metrics["rmse_censor_cap"],
        },
    }
    (outputs_path / "heldout_metrics_summary.json").write_text(
        json.dumps(heldout_summary, indent=2, sort_keys=True)
    )

    gisaid_csv = Path(args.gisaid_csv) if args.gisaid_csv else storage_path / "data" / "PLANT_epiflu_human_241212.csv"
    if not args.skip_gisaid and gisaid_csv.exists():
        print("Embedding GISAID/all-sequence CSV...")
        gisaid_df = pd.read_csv(gisaid_csv)
        if "seq" not in gisaid_df.columns:
            raise ValueError("--gisaid-csv must contain a 'seq' column.")
        gisaid_df = gisaid_df[
            ~gisaid_df["seq"].str.contains("X", regex=False, na=False)
            & ~gisaid_df["seq"].str.contains("B", regex=False, na=False)
            & ~gisaid_df["seq"].str.contains("*", regex=False, na=False)
        ].copy()
        gisaid_df = gisaid_df[gisaid_df["seq"].str.len() == args.max_length].reset_index(drop=True)
        encodes_gisaid = tokenize_sequences(gisaid_df["seq"].tolist(), tokenizer, args.max_length)
        dataset_gisaid = TextDataset(encodes_gisaid)
        dataloader_gisaid = DataLoader(dataset_gisaid, batch_size=64, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer.model.to(device)
        if bf16:
            trainer.model.bfloat16()
        elif fp16:
            trainer.model.half()
        coords = embed_sequences(
            trainer.model,
            dataloader_gisaid,
            use_bf16=bf16,
            use_fp16=fp16,
        )
        gisaid_df["z1"] = coords[:, 0]
        gisaid_df["z2"] = coords[:, 1]
        gisaid_df["z3"] = coords[:, 2]
        gisaid_out = outputs_path / f"{gisaid_csv.stem}_with_coords.csv"
        gisaid_df.to_csv(gisaid_out, index=False)
        print(f"Saved: {gisaid_out}")

    print(f"Saved held-out summaries: {outputs_path / 'heldout_metrics_summary.json'}")
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

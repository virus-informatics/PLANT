#!/usr/bin/env python
"""Train PLANT using the reusable ``plant`` Python module.

Run from the repository root, for example:

python Training/PLANT_train_with_module.py \
  --prefix full_module_train \
  --directory . \
  --batch-size 16 \
  --num-steps 20000 \
  --model facebook/esm2_t33_650M_UR50D
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
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
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


def write_prediction_correlations(df: pd.DataFrame, output_path: Path) -> None:
    """Write Pearson and Spearman correlations for all prediction columns."""
    prediction_columns = [
        "predicted_dist",
        "predicted_dist_censor_cap",
        "predicted_dist_cartography",
        "predicted_dist_cartography_censor_cap",
    ]
    lines = []
    for col in prediction_columns:
        if col not in df.columns:
            continue
        pearson_r, pearson_p, pearson_n = _safe_correlation(df["score"], df[col], method="pearson")
        spearman_r, spearman_p, spearman_n = _safe_correlation(df["score"], df[col], method="spearman")
        lines.extend(
            [
                f"[{col}]",
                f"pearson_r\t{pearson_r:.8g}",
                f"pearson_p\t{pearson_p:.8g}",
                f"pearson_n\t{pearson_n}",
                f"spearman_r\t{spearman_r:.8g}",
                f"spearman_p\t{spearman_p:.8g}",
                f"spearman_n\t{spearman_n}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines))


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
    selected_df_train_val, selected_df_test = split_dataframe_with_stratified_group_kfold(
        selected_df, train_ratio=0.8, seed=args.random_seed
    )
    selected_df_test = selected_df_test.copy()
    selected_df_test["weight"] = 1.0
    selected_df_train_val = add_density_weights(selected_df_train_val)
    selected_df_train, selected_df_val = split_dataframe_with_stratified_group_kfold(
        selected_df_train_val, train_ratio=8 / 9, seed=args.random_seed
    )

    print(
        json.dumps(
            {
                "train": len(selected_df_train),
                "validation": len(selected_df_val),
                "test": len(selected_df_test),
            },
            indent=2,
        )
    )

    print("Loading virus-only sequence pool...")
    sequence_pool_df = load_sequence_pool(sequence_pool_fasta, sequence_pool_metadata)
    sequence_pool_df = sequence_pool_df[sequence_pool_df["seq"].str.len() == args.max_length].reset_index(drop=True)
    print(f"Virus-only sequence pool size: {len(sequence_pool_df)}")

    print("Tokenizing sequences...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset_train = make_paired_dataset(selected_df_train, tokenizer, args.max_length)
    dataset_val = make_paired_dataset(selected_df_val, tokenizer, args.max_length)
    dataset_test = make_paired_dataset(selected_df_test, tokenizer, args.max_length)
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
        load_best_model_at_end=True,
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
        }
    )
    (outputs_path / "training_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True)
    )

    print("Predicting held-out test data...")
    predictions = trainer.predict(dataset_test)
    logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    observed_distance = logits[:, 0]
    cartography_distance = logits[:, 1]
    selected_df_test = selected_df_test.reset_index(drop=True)
    selected_df_test["predicted_dist"] = observed_distance
    selected_df_test["predicted_dist_cartography"] = cartography_distance
    apply_censor_cap(selected_df_test, "censor", "predicted_dist", "score", "predicted_dist_censor_cap")
    apply_censor_cap(
        selected_df_test,
        "censor",
        "predicted_dist_cartography",
        "score",
        "predicted_dist_cartography_censor_cap",
    )

    if isinstance(predictions.predictions, tuple) and len(predictions.predictions) > 1:
        latent = predictions.predictions[1]
        selected_df_test[["z1", "z2", "z3"]] = latent[:, :3]

    test_csv = outputs_path / "test_df_full.csv"
    selected_df_test.to_csv(test_csv, index=False)
    corr_file = outputs_path / "test_prediction_correlations.txt"
    write_prediction_correlations(selected_df_test, corr_file)
    corr, p_value, corr_n = _safe_correlation(
        selected_df_test["score"],
        selected_df_test["predicted_dist_censor_cap"],
        method="pearson",
    )
    print(f"Pearson correlation: {corr:.4f} (p-value: {p_value:.4g}, n={corr_n})")
    print(f"Saved correlations: {corr_file}")

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

    print(f"Saved: {test_csv}")
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

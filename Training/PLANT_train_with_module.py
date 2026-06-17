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
        help="Optional CSV whose seq column will be embedded after training.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--max-length", default=329, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--embedding-batch-size", default=128, type=int)
    parser.add_argument("--num-steps", default=20000, type=int)
    parser.add_argument("--random-seed", default=42, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--reg-weight-decay", default=0.01, type=float)
    parser.add_argument("--max-saves", default=1, type=int)
    parser.add_argument("--save-steps", default=1000, type=int)
    parser.add_argument("--eval-steps", default=1000, type=int)
    parser.add_argument("--warmup-ratio", default=0.1, type=float)
    parser.add_argument("--num-samples-per-combination", default=1, type=int)
    parser.add_argument("--CSE-w", default=0.0, type=float)
    parser.add_argument("--CSE-w-virus-only", default=0.0, type=float)
    parser.add_argument("--semantic-w", default=0.2, type=float)
    parser.add_argument("--semantic-w-virus-only", default=0.2, type=float)
    parser.add_argument("--cart-w", default=0.05, type=float)
    parser.add_argument("--dropout-regressor", default=0.05, type=float)
    parser.add_argument("--reg-intermediate-dim", default=256, type=int)
    parser.add_argument("--CSE-alpha", default=0.0, type=float)
    parser.add_argument("--intermediate-dim-encoder", default=64, type=int)
    parser.add_argument("--dropout-encoder", default=0.1, type=float)
    parser.add_argument("--lg-w", default=0.01, type=float)
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


def main() -> None:
    start = time.time()
    args = parse_args()
    set_seed(args.random_seed)

    storage_path = Path(args.directory).resolve()
    outputs_path = storage_path / "Season_based_split_performance" / args.prefix / "trained_until_full"
    outputs_path.mkdir(parents=True, exist_ok=True)

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
    ohe_vp = make_one_hot_encoder(selected_df_train["virus_passage_category"])
    ohe_rp = make_one_hot_encoder(selected_df_train["reference_passage_category"])
    set_encoders(ohe_virus, ohe_ref, ohe_vp, ohe_rp)

    joblib.dump(ohe_virus, outputs_path / "virus_encoder.joblib")
    joblib.dump(ohe_ref, outputs_path / "ref_encoder.joblib")
    joblib.dump(ohe_vp, outputs_path / "vp_encoder.joblib")
    joblib.dump(ohe_rp, outputs_path / "rp_encoder.joblib")

    effects_len = len(ohe_ref.categories_[0]) + len(ohe_vp.categories_[0]) + len(ohe_rp.categories_[0])
    virus_effects_len = len(ohe_virus.categories_[0])

    print("Estimating ESM embedding scale factor...")
    embed_scale_factor = estimate_embed_scale_factor(
        dataset_train,
        esm_model_name=args.model,
        batch_size=args.embedding_batch_size,
        use_fp16=not args.no_fp16,
    )
    print(f"embed_scale_factor: {embed_scale_factor:.6g}")

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
    )

    optimizer = build_plant_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regressor_weight_decay=args.reg_weight_decay,
    )

    fp16 = (not args.no_fp16) and torch.cuda.is_available()
    training_args = make_training_args(
        output_dir=str(outputs_path / "results"),
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=1,
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
    corr, p_value = scipy.stats.pearsonr(
        selected_df_test["score"], selected_df_test["predicted_dist_censor_cap"]
    )
    print(f"Pearson correlation: {corr:.4f} (p-value: {p_value:.4g})")

    if args.gisaid_csv:
        print("Embedding additional GISAID/all-sequence CSV...")
        gisaid_df = pd.read_csv(args.gisaid_csv)
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
        if fp16:
            trainer.model.half()
        coords = embed_sequences(trainer.model, dataloader_gisaid, use_fp16=fp16)
        gisaid_df["z1"] = coords[:, 0]
        gisaid_df["z2"] = coords[:, 1]
        gisaid_df["z3"] = coords[:, 2]
        gisaid_out = outputs_path / f"{Path(args.gisaid_csv).stem}_with_coords.csv"
        gisaid_df.to_csv(gisaid_out, index=False)
        print(f"Saved: {gisaid_out}")

    print(f"Saved: {test_csv}")
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

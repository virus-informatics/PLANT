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
from functools import partial
import json
import random
import re
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
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EsmConfig,
    TrainingArguments,
)

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
    semanticESM,
    tokenize_sequences,
)


def str2bool(value) -> bool:
    """Parse common string forms of booleans for argparse."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value such as true/false, got {value!r}."
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
    parser.add_argument(
        "--max-length",
        default=329,
        type=int,
        help=(
            "Expected raw amino-acid sequence length. The tokenizer length is "
            "set internally to max_length + 2 for ESM <cls>/<eos> tokens."
        ),
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", default=16, type=int)
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", default=32, type=int)
    parser.add_argument("--embedding-batch-size", "--embedding_batch_size", dest="embedding_batch_size", default=128, type=int)
    parser.add_argument("--num-steps", "--num_steps", dest="num_steps", default=20000, type=int)
    parser.add_argument("--random-seed", "--random_seed", dest="random_seed", default=42, type=int)
    parser.add_argument(
        "--split-seed",
        "--split_seed",
        dest="split_seed",
        default=None,
        type=int,
        help=(
            "Seed for the train/validation/test split and the virus-only pool "
            "subsampling. Defaults to --random-seed. Fix this while varying "
            "--random-seed to measure training-only run-to-run variance on an "
            "identical data split."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        default=1e-4,
        type=float,
        help=(
            "Backward-compatible shared learning-rate fallback. Component-specific "
            "rates below override it when supplied."
        ),
    )
    parser.add_argument(
        "--encoder-learning-rate",
        "--encoder_learning_rate",
        dest="encoder_learning_rate",
        default=3e-5,
        type=float,
        help=(
            "Learning rate for trainable ESM encoder parameters (LoRA or full fine-tuning). "
            "Default: 3e-5."
        ),
    )
    parser.add_argument(
        "--regressor-learning-rate",
        "--regressor_learning_rate",
        dest="regressor_learning_rate",
        default=3e-4,
        type=float,
        help=(
            "Learning rate for the antigenic-map regressor. Default: 3e-4."
        ),
    )
    parser.add_argument(
        "--auxiliary-learning-rate",
        "--auxiliary_learning_rate",
        dest="auxiliary_learning_rate",
        default=1e-4,
        type=float,
        help=(
            "Learning rate for systematic-error heads, reference transforms, and "
            "embed_scale. Default: 1e-4."
        ),
    )
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", default=0.01, type=float)
    parser.add_argument("--reg-weight-decay", "--reg_weight_decay", dest="reg_weight_decay", default=0.01, type=float)
    parser.add_argument("--max-saves", "--max_saves", dest="max_saves", default=1, type=int)
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", default=1000, type=int)
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", default=1000, type=int)
    parser.add_argument(
        "--selection-mae-lambda",
        "--selection_mae_lambda",
        dest="selection_mae_lambda",
        default=1.0,
        type=float,
        help=(
            "Lambda in the maximized validation score: censor-cap Pearson + "
            "censor-cap Spearman - lambda * censor-cap MAE."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        dest="early_stopping_patience",
        default=5,
        type=int,
        help=(
            "Number of evaluations without improvement before stopping. "
            "Set to 0 to disable early stopping."
        ),
    )
    parser.add_argument(
        "--early-stopping-threshold",
        "--early_stopping_threshold",
        dest="early_stopping_threshold",
        default=0.001,
        type=float,
        help="Minimum validation-score improvement counted by early stopping.",
    )
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num-samples-per-combination", "--num_samples_per_combination", dest="num_samples_per_combination", default=1, type=int)
    parser.add_argument("--CSE-w", "--CSE_w", dest="CSE_w", default=0.0, type=float)
    parser.add_argument(
        "--CSE-w-virus-only",
        "--CSE_w_virus_only",
        dest="CSE_w_virus_only",
        default=None,
        type=float,
        help=(
            "Deprecated compatibility option; ignored. Unified CSE uses --CSE-w "
            "for paired viruses, references, and virus-only sequences."
        ),
    )
    parser.add_argument("--semantic-w", "--semantic_w", dest="semantic_w", default=0.1, type=float)
    parser.add_argument(
        "--semantic-w-virus-only",
        "--semantic_w_virus_only",
        dest="semantic_w_virus_only",
        default=None,
        type=float,
        help=(
            "Deprecated compatibility option; ignored. Unified semantic loss uses "
            "--semantic-w for paired viruses, references, and virus-only sequences."
        ),
    )
    parser.add_argument("--cart-w", "--cart_w", dest="cart_w", default=0.1, type=float)
    parser.add_argument("--dropout-regressor", "--dropout_regressor", dest="dropout_regressor", default=0.05, type=float)
    parser.add_argument("--reg-intermediate-dim", "--reg_intermediate_dim", dest="reg_intermediate_dim", default=256, type=int)
    parser.add_argument("--CSE-alpha", "--CSE_alpha", dest="CSE_alpha", default=0.0, type=float)
    parser.add_argument("--intermediate-dim-encoder", "--intermediate_dim_encoder", dest="intermediate_dim_encoder", default=64, type=int)
    parser.add_argument("--dropout-encoder", "--dropout_encoder", dest="dropout_encoder", default=0.1, type=float)
    parser.add_argument("--lg-w", "--lg_w", dest="lg_w", default=0.0, type=float)

    systematic_error_group = parser.add_mutually_exclusive_group()
    systematic_error_group.add_argument(
        "--use-systematic-error",
        "--use_systematic_error",
        dest="use_systematic_error",
        action="store_true",
        help=(
            "Include learned virus/reference/passage systematic-error corrections "
            "in observed-distance prediction (default)."
        ),
    )
    systematic_error_group.add_argument(
        "--no-use-systematic-error",
        "--no_use_systematic_error",
        "--no-systematic-error",
        "--no_systematic_error",
        dest="use_systematic_error",
        action="store_false",
        help=(
            "Ablation mode: exclude systematic-error corrections so that "
            "observed_distance equals the cartographic distance."
        ),
    )
    parser.set_defaults(use_systematic_error=True)

    parser.add_argument(
        "--reference-transform-mode",
        "--reference_transform_mode",
        dest="reference_transform_mode",
        choices=["none", "full", "diagonal"],
        default="none",
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
    freeze_esm_group = parser.add_mutually_exclusive_group()
    freeze_esm_group.add_argument(
        "--freeze-esm",
        "--freeze_esm",
        dest="freeze_esm",
        nargs="?",
        const=True,
        type=str2bool,
        help=(
            "Freeze the ESM model completely (no LoRA, no full fine-tuning). "
            "Must be used together with --no-use-lora explicitly "
            "(e.g., --freeze-esm --no-use-lora). "
            "Accepts optional true/false for backward-compatible CLI usage."
        ),
    )
    freeze_esm_group.add_argument(
        "--no-freeze-esm",
        "--no_freeze_esm",
        dest="freeze_esm",
        action="store_false",
        help="Do not freeze the ESM model (default).",
    )
    parser.set_defaults(freeze_esm=False)
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument(
        "--use-lora",
        "--use_lora",
        dest="use_lora",
        nargs="?",
        const=True,
        type=str2bool,
        help=(
            "Use LoRA adapters for the ESM model. This is the default. "
            "Accepts optional true/false for backward-compatible CLI usage."
        ),
    )
    lora_group.add_argument(
        "--no-use-lora",
        "--no_use_lora",
        dest="use_lora",
        action="store_false",
        help=(
            "Disable LoRA adapters. Without --freeze-esm, this enables full fine-tuning "
            "of the ESM model. With --freeze-esm, the ESM model is completely frozen."
        ),
    )
    parser.set_defaults(use_lora=True)
    parser.add_argument(
        "--lora-r",
        "--lora_r",
        dest="lora_r",
        default=16,
        type=int,
        help="Rank for the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-alpha",
        "--lora_alpha",
        dest="lora_alpha",
        default=32,
        type=int,
        help="Alpha for the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-dropout",
        "--lora_dropout",
        dest="lora_dropout",
        default=0.1,
        type=float,
        help="Dropout for the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-target-modules",
        "--lora_target_modules",
        dest="lora_target_modules",
        type=str,
        default=None,
        help=(
            "Comma-separated list of module names to apply LoRA adapters to. "
        ),
    )
    parser.add_argument(
        "--lora-bias",
        "--lora_bias",
        dest="lora_bias",
        default="none",
        choices=["none", "all", "lora_only"],
        type=str,
        help=(
            "Bias option for LoRA adapters. 'none' means no bias, 'all' means "
            "all biases are trainable, and 'lora_only' means only LoRA biases "
            "are trainable."
        )
    )
    parser.add_argument(
        "--split-mode",
        "--split_mode",
        dest="split_mode",
        choices=["season", "full"],
        default="season",
        help=(
            "How to construct train/validation/test/future splits. "
            "'season' uses --cutoff-season to hold out later seasons as a future set. "
            "'full' uses all seasons for the random train/validation/test split and "
            "does not create or evaluate a future dataset."
        ),
    )
    parser.add_argument(
        "--cutoff-season",
        "--cutoff_season",
        dest="cutoff_season",
        default="NH2022",
        help=(
            "Cutoff season for past/future evaluation split based on the paired-data "
            "season column. Used only when --split-mode season. Examples: NH2014 or "
            "SH2014. Rows with season <= cutoff are used for the random "
            "train/validation/test split; later seasons are held out as the future "
            "dataset. Season order is ... SH2013, NH2014, SH2014, NH2015, ..."
        ),
    )
    parser.add_argument(
        "--season-col",
        "--season_col",
        dest="season_col",
        default="season",
        help="Column name in the paired CSV containing season labels such as NH2014 or SH2014.",
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

    args = parser.parse_args()

    if args.freeze_esm and args.use_lora:
        parser.error(
            "--freeze-esm and --use-lora cannot be set simultaneously. "
            "Please specify --no-use-lora explicitly when using --freeze-esm."
        )
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0.")
    for option_name in (
        "encoder_learning_rate",
        "regressor_learning_rate",
        "auxiliary_learning_rate",
    ):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be > 0.")
    if args.selection_mae_lambda < 0:
        parser.error("--selection-mae-lambda must be >= 0.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0.")
    if args.early_stopping_threshold < 0:
        parser.error("--early-stopping-threshold must be >= 0.")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error("--eval-steps and --save-steps must both be > 0.")
    if args.save_steps % args.eval_steps != 0:
        parser.error(
            "With load_best_model_at_end=True, --save-steps must be an integer "
            "multiple of --eval-steps."
        )

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_lora_target_modules(value: str | None) -> list[str]:
    """Parse comma-separated LoRA module names.

    If None is given, defaults to ["query", "key", "value"] as the best-known setting.
    An empty string is not allowed and will raise a ValueError.
    """
    if value is None:
        return ["query", "key", "value"]
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise ValueError(
            "--lora-target-modules cannot be empty. "
            "Specify at least one module name (e.g., 'query,key,value'), "
            "or omit the argument to use the default ['query', 'key', 'value']."
        )
    return modules


CATEGORY_MAPPING_COLUMNS = (
    "date",
    "virus",
    "reference",
    "virus_passage",
    "reference_passage",
)


def build_category_mappings(
    df: pd.DataFrame,
    *,
    category_cols: tuple[str, ...] = CATEGORY_MAPPING_COLUMNS,
) -> dict[str, object]:
    """Build string-to-integer mappings used before one-hot encoding.

    The training pipeline first converts metadata strings such as virus_passage
    to integer ``*_category`` columns with pandas categorical codes.  The
    OneHotEncoder objects are then fitted on those integer codes, not on the
    original strings.  This JSON-serializable mapping is therefore required to
    reproduce systematic-error inference on new data.
    """
    mappings: dict[str, object] = {
        "__metadata__": {
            "format_version": 1,
            "description": (
                "Original metadata value -> integer *_category mappings used "
                "before fitting the OneHotEncoder systematic-error encoders."
            ),
            "unknown_category": -1,
            "category_columns": list(category_cols),
        },
        "__category_to_value__": {},
    }

    reverse_mappings: dict[str, dict[str, str]] = {}
    for col in category_cols:
        category_col = f"{col}_category"
        if col not in df.columns or category_col not in df.columns:
            raise ValueError(
                f"Cannot build category mapping: missing {col!r} or {category_col!r}."
            )

        mapping_df = (
            df[[col, category_col]]
            .drop_duplicates()
            .sort_values([category_col, col], kind="mergesort")
            .reset_index(drop=True)
        )

        value_to_category: dict[str, int] = {}
        category_to_value: dict[str, str] = {}
        for _, row in mapping_df.iterrows():
            value = str(row[col])
            category = int(row[category_col])

            # Defensive checks: pandas categorical codes should be one-to-one for
            # a single column.  If this ever fails, the saved mapping would be
            # unsafe for inference.
            if value in value_to_category and value_to_category[value] != category:
                raise ValueError(
                    f"Inconsistent category mapping for column {col!r}, "
                    f"value {value!r}: {value_to_category[value]} vs {category}"
                )
            category_key = str(category)
            if category_key in category_to_value and category_to_value[category_key] != value:
                raise ValueError(
                    f"Inconsistent reverse category mapping for column {col!r}, "
                    f"category {category}: {category_to_value[category_key]!r} vs {value!r}"
                )

            value_to_category[value] = category
            category_to_value[category_key] = value

        mappings[col] = value_to_category
        reverse_mappings[col] = category_to_value

    mappings["__category_to_value__"] = reverse_mappings
    return mappings


def save_category_mappings(df: pd.DataFrame, output_path: Path) -> dict[str, object]:
    """Save metadata string -> integer category mappings for inference."""
    category_mappings = build_category_mappings(df)
    output_path.write_text(
        json.dumps(category_mappings, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return category_mappings


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

    Backward-compatible wrapper around ``resolve_partial_date_interval_for_time_split``.
    For season splitting, callers should usually use the full interval so that
    ambiguous dates near the cutoff are not forced into the past set.
    """
    _earliest, latest = resolve_partial_date_interval_for_time_split(date_value)
    return latest


def resolve_partial_date_interval_for_time_split(date_value) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the earliest and latest plausible dates for a partial/invalid date.

    The paired dataset can contain partially specified or non-standard assay dates
    such as ``1993-aaa``.  For conservative past/future splitting, these values are
    interpreted as date ranges rather than single dates.

    Examples:
    - ``1993-aaa`` -> 1993-01-01 to 1993-12-31
    - ``2024-01-aaa`` -> 2024-01-01 to 2024-01-31
    - ``2024-01`` -> 2024-01-01 to 2024-01-31
    - ``2024-01-15`` -> 2024-01-15 to 2024-01-15
    """
    if pd.isna(date_value):
        return pd.NaT, pd.NaT

    # CSV columns containing bare years plus missing values may be inferred as
    # floating point, turning 1996 into 1996.0. Recover integer-valued numeric
    # years before string parsing so they are treated as full-year intervals.
    if isinstance(date_value, (int, np.integer)):
        raw = str(int(date_value))
    elif isinstance(date_value, (float, np.floating)):
        if not np.isfinite(date_value):
            return pd.NaT, pd.NaT
        raw = str(int(date_value)) if float(date_value).is_integer() else str(date_value)
    else:
        raw = str(date_value).strip()

    if not raw:
        return pd.NaT, pd.NaT

    # Prefer explicit year-first strings because pandas interprets bare years
    # as January 1, while for splitting they should represent the whole year.
    match = re.match(r"^\s*(\d{4})(?:[-/](.*))?\s*$", raw)
    if match:
        year = int(match.group(1))
        remainder = match.group(2)

        if year < 1:
            return pd.NaT, pd.NaT

        if remainder is None or remainder == "":
            return (
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31),
            )

        parts = re.split(r"[-/]", remainder)
        month_token = parts[0] if len(parts) >= 1 else ""
        if not month_token.isdigit():
            return (
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31),
            )

        month = int(month_token)
        if not 1 <= month <= 12:
            return (
                pd.Timestamp(year=year, month=1, day=1),
                pd.Timestamp(year=year, month=12, day=31),
            )

        last_day = calendar.monthrange(year, month)[1]
        day_token = parts[1] if len(parts) >= 2 else ""
        if not day_token.isdigit():
            return (
                pd.Timestamp(year=year, month=month, day=1),
                pd.Timestamp(year=year, month=month, day=last_day),
            )

        day = int(day_token)
        if not 1 <= day <= last_day:
            return (
                pd.Timestamp(year=year, month=month, day=1),
                pd.Timestamp(year=year, month=month, day=last_day),
            )

        exact = pd.Timestamp(year=year, month=month, day=day)
        return exact, exact

    # Fallback for non-year-first valid dates such as "09/02/2010" or "Nov-2009".
    try:
        exact = pd.to_datetime(raw, errors="raise")
    except Exception:
        return pd.NaT, pd.NaT

    if pd.isna(exact):
        return pd.NaT, pd.NaT
    exact = pd.Timestamp(exact).normalize()
    return exact, exact


def cutoff_season_end_date(cutoff_season: str) -> pd.Timestamp:
    """Return the inclusive end date of a cutoff season.

    This follows the extractor's season convention:
    SHYYYY = YYYY-02-01 to YYYY-08-31
    NHYYYY = (YYYY-1)-09-01 to YYYY-01-31
    """
    normalized, _order = parse_season_label(cutoff_season)
    hemisphere = normalized[:2]
    year = int(normalized[2:])

    if hemisphere == "SH":
        return pd.Timestamp(year=year, month=8, day=31)
    if hemisphere == "NH":
        return pd.Timestamp(year=year, month=1, day=31)

    raise ValueError(f"Unknown hemisphere in cutoff season: {cutoff_season!r}")


def safe_season_order_to_label(order) -> str | None:
    """Convert a season order to a label, returning None for missing values."""
    if pd.isna(order):
        return None
    return season_order_to_label(int(order))


def sorted_valid_season_labels(values) -> list[str]:
    """Sort parseable non-missing season labels according to PLANT season order."""
    labels = []
    for value in values:
        if pd.isna(value):
            continue
        try:
            parse_season_label(value)
        except ValueError:
            continue
        labels.append(str(value))
    return sorted(set(labels), key=lambda s: parse_season_label(s)[1])


_SEASON_RE_NH_SH_FIRST = re.compile(r"^\s*(NH|SH)\s*[-_/ ]?\s*(\d{4})\s*$", re.IGNORECASE)
_SEASON_RE_YEAR_FIRST = re.compile(r"^\s*(\d{4})\s*[-_/ ]?\s*(NH|SH)\s*$", re.IGNORECASE)


def parse_season_label(season_value) -> tuple[str, int]:
    """Parse a season label and return a normalized label plus sortable order.

    The order follows the antigenic-assay convention requested for PLANT:
    ``... SH2013 < NH2014 < SH2014 < NH2015 ...``. Thus ``NHYYYY`` is encoded
    as ``YYYY * 2`` and ``SHYYYY`` as ``YYYY * 2 + 1``.
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
    """Convert a sortable season order back to a normalized season label."""
    order = int(order)
    if order % 2 == 0:
        return f"NH{order // 2:04d}"
    return f"SH{(order - 1) // 2:04d}"


def add_season_order_columns(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Add normalized season and sortable season-order columns.

    By default, unparseable/missing season labels are kept with NaN season_order.
    This is important for ``--split-mode full`` and for date-based fallback in
    ``--split-mode season``.
    """
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
            parsed_orders.append(float(order))

    out["season_normalized"] = parsed_labels
    out["season_order"] = parsed_orders

    invalid = out["season_order"].isna()
    if invalid.any():
        examples = sorted({str(v) for v in invalid_values})[:10]
        action = "Dropping" if drop_invalid else "Keeping"
        print(
            f"{action} rows with unparseable/missing season values: "
            f"{int(invalid.sum())}. Examples: {examples}"
        )
        if drop_invalid:
            out = out.loc[~invalid].copy()

    if drop_invalid:
        out["season_order"] = out["season_order"].astype(int)
    return out


def split_past_future_by_cutoff_season(
    df: pd.DataFrame,
    *,
    cutoff_season: str,
    season_col: str = "season",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split paired data into past/future sets using season labels plus date fallback.

    Rules:
    1. If ``season_col`` is parseable, use the ordered NH/SH season label.
    2. If season is missing/unparseable, infer a conservative date interval from
       ``date_col``.  Assign to past only when the latest plausible date is on or
       before the cutoff season end date.  Assign to future only when the earliest
       plausible date is after the cutoff season end date.
    3. Rows that remain ambiguous are returned separately and excluded from
       past/future training/evaluation to avoid leakage.
    """
    cutoff_normalized, cutoff_order = parse_season_label(cutoff_season)
    cutoff_end = cutoff_season_end_date(cutoff_normalized)
    out = add_season_order_columns(df, season_col=season_col, drop_invalid=False)

    out["season_split_source"] = pd.NA
    out["date_interval_start"] = pd.NaT
    out["date_interval_end"] = pd.NaT
    out["season_split_status"] = pd.NA

    known_season = out["season_order"].notna()
    past_mask = known_season & (out["season_order"] <= cutoff_order)
    future_mask = known_season & (out["season_order"] > cutoff_order)

    out.loc[past_mask | future_mask, "season_split_source"] = "season"
    out.loc[past_mask, "season_split_status"] = "past"
    out.loc[future_mask, "season_split_status"] = "future"

    unknown_season = ~known_season
    if unknown_season.any():
        if date_col not in out.columns:
            raise ValueError(
                f"Cannot use date fallback for season split: missing date column {date_col!r}."
            )

        intervals = out.loc[unknown_season, date_col].apply(resolve_partial_date_interval_for_time_split)
        starts = intervals.apply(lambda x: x[0])
        ends = intervals.apply(lambda x: x[1])
        out.loc[unknown_season, "date_interval_start"] = starts
        out.loc[unknown_season, "date_interval_end"] = ends

        date_known = starts.notna() & ends.notna()
        date_past = unknown_season.copy()
        date_past.loc[:] = False
        date_future = unknown_season.copy()
        date_future.loc[:] = False

        date_past.loc[starts.index] = date_known & (ends <= cutoff_end)
        date_future.loc[starts.index] = date_known & (starts > cutoff_end)

        out.loc[date_past, "season_split_source"] = "date"
        out.loc[date_past, "season_split_status"] = "past"
        out.loc[date_future, "season_split_source"] = "date"
        out.loc[date_future, "season_split_status"] = "future"

        n_unknown = int(unknown_season.sum())
        n_date_past = int(date_past.sum())
        n_date_future = int(date_future.sum())
        n_ambiguous = int(n_unknown - n_date_past - n_date_future)
        print(
            "Season split date fallback for rows with missing/unparseable season: "
            f"unknown={n_unknown}, date_past={n_date_past}, "
            f"date_future={n_date_future}, ambiguous_or_unparseable={n_ambiguous}, "
            f"cutoff_end={cutoff_end.date()}"
        )

    ambiguous = out.loc[out["season_split_status"].isna()].copy()
    past = out.loc[out["season_split_status"] == "past"].copy()
    future = out.loc[out["season_split_status"] == "future"].copy()

    if not ambiguous.empty:
        examples = (
            ambiguous[[season_col, date_col]]
            .head(10)
            .astype("object")
            .where(pd.notna(ambiguous[[season_col, date_col]].head(10)), None)
            .to_dict("records")
        )
        print(
            "Excluding rows with ambiguous season/date for season split: "
            f"{len(ambiguous)}. Examples: {examples}"
        )

    if past.empty:
        raise ValueError(
            f"No rows were assigned to the past set using cutoff_season={cutoff_normalized!r}."
        )
    if future.empty:
        raise ValueError(
            f"No rows were assigned to the future set using cutoff_season={cutoff_normalized!r}."
        )

    return (
        past.reset_index(drop=True),
        future.reset_index(drop=True),
        ambiguous.reset_index(drop=True),
    )


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


def filter_sequence_pool_by_cutoff_date(
    sequence_pool_df: pd.DataFrame,
    *,
    cutoff_season: str,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude virus-only sequences that may occur after the season cutoff.

    Partial or malformed year-first dates are interpreted as intervals with
    :func:`resolve_partial_date_interval_for_time_split`. For example, ``1996--``
    represents 1996-01-01 through 1996-12-31. A sequence is retained only when
    its latest plausible collection date is on or before the cutoff end date.
    Missing, unparseable, or cutoff-spanning dates are excluded conservatively.

    This function must be applied before virus-only downsampling.
    """
    if date_col not in sequence_pool_df.columns:
        raise ValueError(
            "Virus-only metadata must contain a collection-date column for "
            f"season-based training: missing {date_col!r}."
        )

    cutoff_normalized, _ = parse_season_label(cutoff_season)
    cutoff_end = cutoff_season_end_date(cutoff_normalized)
    out = sequence_pool_df.copy()

    intervals = out[date_col].apply(resolve_partial_date_interval_for_time_split)
    out["date_interval_start"] = intervals.apply(lambda value: value[0])
    out["date_interval_end"] = intervals.apply(lambda value: value[1])

    date_known = out["date_interval_start"].notna() & out["date_interval_end"].notna()
    keep_mask = date_known & (out["date_interval_end"] <= cutoff_end)
    definitely_future = date_known & (out["date_interval_start"] > cutoff_end)
    ambiguous_or_unparseable = ~(keep_mask | definitely_future)

    out["cutoff_filter_status"] = "ambiguous_or_unparseable"
    out.loc[keep_mask, "cutoff_filter_status"] = "past"
    out.loc[definitely_future, "cutoff_filter_status"] = "future"

    kept = out.loc[keep_mask].copy().reset_index(drop=True)
    excluded = out.loc[~keep_mask].copy().reset_index(drop=True)

    print(
        "Virus-only cutoff-date filter: "
        f"cutoff={cutoff_normalized}, cutoff_end={cutoff_end.date()}, "
        f"input={len(out)}, retained_past={int(keep_mask.sum())}, "
        f"excluded_future={int(definitely_future.sum())}, "
        f"excluded_ambiguous_or_unparseable={int(ambiguous_or_unparseable.sum())}"
    )

    if kept.empty:
        raise ValueError(
            "Virus-only sequence pool is empty after cutoff-date filtering: "
            f"cutoff_season={cutoff_normalized!r}."
        )

    return kept, excluded


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


def clean_paired_dataset(
    path: Path,
    max_length: int,
    seed: int,
    *,
    season_col: str = "season",
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
        #"reference_collection_date",
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

    # Keep rows with missing/unparseable season so that --split-mode full can use
    # them and --split-mode season can rescue clearly past/future rows by date.
    nonnull_required_cols = [col for col in required_cols if col != "season"]
    selected = df[required_cols].dropna(subset=nonnull_required_cols).copy()
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
    stratify_labels = (df["virus_collection_year"].astype(str) + "_"
                     + df["virus_passage"].astype(str) + "_"
                     + df["reference_passage"].astype(str))
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


def compute_metrics_for_trainer(
    eval_pred,
    *,
    selection_mae_lambda: float = 1.0,
) -> dict[str, float]:
    """Compute validation metrics and the maximized checkpoint-selection score."""
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

    metrics = compute_prediction_metrics(
        labels,
        censors,
        {
            "predicted_dist": observed_distance,
            "predicted_dist_censor_cap": predicted_dist_censor_cap,
            "predicted_dist_cartography": cartography_distance,
            "predicted_dist_cartography_censor_cap": predicted_dist_cartography_censor_cap,
        },
    )

    components = np.asarray(
        [
            metrics["pearson_censor_cap"],
            metrics["spearman_censor_cap"],
            metrics["mae_censor_cap"],
        ],
        dtype=float,
    )
    if np.all(np.isfinite(components)):
        selection_score = (
            components[0]
            + components[1]
            - float(selection_mae_lambda) * components[2]
        )
    else:
        # NaN checkpoint metrics can prevent Trainer from identifying a best model
        # and can stall early stopping. Keep the original component metrics as NaN
        # for diagnosis, but assign an unambiguously poor finite selection score.
        selection_score = -1.0e9

    metrics["selection_score_censor_cap"] = float(selection_score)
    return metrics


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



def _json_safe_value(value):
    """Convert numpy scalars and non-finite floats into JSON-friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def compute_prediction_metrics_by_season(
    df: pd.DataFrame,
    *,
    season_col: str = "season_normalized",
) -> pd.DataFrame:
    """Compute prediction metrics separately for each future season."""
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


def write_future_metrics_by_season(
    df: pd.DataFrame,
    outputs_path: Path,
    *,
    season_col: str = "season_normalized",
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Save future-set metrics computed independently for each season."""
    metrics_df = compute_prediction_metrics_by_season(df, season_col=season_col)

    csv_path = outputs_path / "future_prediction_metrics_by_season.csv"
    metrics_df.to_csv(csv_path, index=False)

    json_ready = {}
    for _, row in metrics_df.iterrows():
        season = row["season"]
        json_ready[str(season)] = {
            str(key): _json_safe_value(value)
            for key, value in row.items()
            if key != "season"
        }

    json_path = outputs_path / "future_prediction_metrics_by_season.json"
    json_path.write_text(json.dumps(json_ready, indent=2, sort_keys=True))

    txt_path = outputs_path / "future_prediction_metrics_by_season.txt"
    lines = []
    for _, row in metrics_df.iterrows():
        lines.append(f"[{row['season']}] n={int(row['n'])}")
        for col in PREDICTION_COLUMNS:
            pearson_key = f"pearson_{col}"
            if pearson_key not in row:
                continue
            lines.extend(
                [
                    f"  {col}",
                    f"    pearson_r	{row[f'pearson_{col}']:.8g}",
                    f"    pearson_n	{int(row[f'pearson_n_{col}'])}",
                    f"    spearman_r	{row[f'spearman_{col}']:.8g}",
                    f"    spearman_n	{int(row[f'spearman_n_{col}'])}",
                    f"    mae	{row[f'mae_{col}']:.8g}",
                    f"    rmse	{row[f'rmse_{col}']:.8g}",
                ]
            )
        lines.append("")
    txt_path.write_text("\n".join(lines))

    print(f"Saved future per-season metrics: {csv_path}")
    print(f"Saved future per-season metrics JSON: {json_path}")
    print(f"Saved future per-season metrics text: {txt_path}")
    return metrics_df, json_ready


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

    # The data split and the virus-only subsample are held fixed by split_seed so
    # that run-to-run variance can be attributed to training stochasticity alone.
    split_seed = args.random_seed if args.split_seed is None else args.split_seed
    print(f"Seeds: random_seed={args.random_seed}, split_seed={split_seed}")

    deprecated_regularization_args = {
        "--CSE-w-virus-only": args.CSE_w_virus_only,
        "--semantic-w-virus-only": args.semantic_w_virus_only,
    }
    supplied_deprecated = {
        name: value
        for name, value in deprecated_regularization_args.items()
        if value is not None
    }
    if supplied_deprecated:
        print(
            "Warning: split virus-only regularization weights are deprecated and "
            f"ignored under unified regularization: {supplied_deprecated}"
        )

    storage_path = Path(args.directory).resolve()
    if args.split_mode == "season":
        output_split_label = f"trained_until_{parse_season_label(args.cutoff_season)[0]}"
    else:
        output_split_label = "full_model"
    outputs_path = storage_path / "Season_based_split_performance" / args.prefix / output_split_label
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
    selected_df = clean_paired_dataset(
        paired_csv,
        args.max_length,
        args.random_seed,
        season_col=args.season_col,
    )
    selected_df_with_season = add_season_order_columns(
        selected_df,
        season_col="season",
        drop_invalid=False,
    )

    if args.split_mode == "season":
        selected_df_past, selected_df_future, selected_df_season_ambiguous = split_past_future_by_cutoff_season(
            selected_df,
            cutoff_season=args.cutoff_season,
            season_col="season",
            date_col="date",
        )
        if not selected_df_season_ambiguous.empty:
            ambiguous_path = outputs_path / "season_split_ambiguous_rows.csv"
            selected_df_season_ambiguous.to_csv(ambiguous_path, index=False)
            print(f"Saved ambiguous season/date rows excluded from season split: {ambiguous_path}")
        has_future = True
    elif args.split_mode == "full":
        selected_df_past = selected_df_with_season.reset_index(drop=True)
        selected_df_future = None
        selected_df_season_ambiguous = pd.DataFrame()
        has_future = False
    else:  # Defensive guard in case argparse choices are bypassed.
        raise ValueError(f"Unknown split_mode: {args.split_mode!r}")

    selected_df_train_val, selected_df_test = split_dataframe_with_stratified_group_kfold(
        selected_df_past, train_ratio=0.9, seed=split_seed
    )
    selected_df_test = selected_df_test.copy()
    selected_df_test["weight"] = 1.0
    if has_future:
        selected_df_future = selected_df_future.copy()
        selected_df_future["weight"] = 1.0

    selected_df_train_val = add_density_weights(selected_df_train_val)
    selected_df_train, selected_df_val = split_dataframe_with_stratified_group_kfold(
        selected_df_train_val, train_ratio=8 / 9, seed=split_seed
    )

    # Save only categories observed in the training split. Validation/test/future-only
    # metadata values are therefore treated as unknown during external inference.
    category_mappings_path = outputs_path / "category_mappings.json"
    category_mappings = save_category_mappings(selected_df_train, category_mappings_path)
    print(f"Saved training-only category mappings: {category_mappings_path}")

    split_summary = {
        "split_mode": args.split_mode,
        "cutoff_season": parse_season_label(args.cutoff_season)[0] if args.split_mode == "season" else None,
        "cutoff_season_end_date": str(cutoff_season_end_date(args.cutoff_season).date()) if args.split_mode == "season" else None,
        "season_col": args.season_col,
        "season_order_convention": "... SH2013 < NH2014 < SH2014 < NH2015 ...",
        "target_random_split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "all_after_cleaning": len(selected_df_with_season),
        "rows_with_missing_or_unparseable_season_after_cleaning": int(selected_df_with_season["season_order"].isna().sum()),
        "season_split_ambiguous_or_unparseable_date": len(selected_df_season_ambiguous),
        "past_total": len(selected_df_past),
        "future": len(selected_df_future) if has_future else 0,
        "train": len(selected_df_train),
        "validation": len(selected_df_val),
        "test": len(selected_df_test),
        "past_min_season": safe_season_order_to_label(selected_df_past["season_order"].min()) if "season_order" in selected_df_past.columns else None,
        "past_max_season": safe_season_order_to_label(selected_df_past["season_order"].max()) if "season_order" in selected_df_past.columns else None,
        "past_seasons": sorted_valid_season_labels(selected_df_past["season_normalized"].unique().tolist()) if "season_normalized" in selected_df_past.columns else [],
        "past_split_sources": selected_df_past["season_split_source"].value_counts(dropna=False).astype(int).to_dict() if "season_split_source" in selected_df_past.columns else {},
    }
    if has_future:
        split_summary.update(
            {
                "future_min_season": safe_season_order_to_label(selected_df_future["season_order"].min()) if "season_order" in selected_df_future.columns else None,
                "future_max_season": safe_season_order_to_label(selected_df_future["season_order"].max()) if "season_order" in selected_df_future.columns else None,
                "future_seasons": sorted_valid_season_labels(selected_df_future["season_normalized"].unique().tolist()) if "season_normalized" in selected_df_future.columns else [],
                "future_split_sources": selected_df_future["season_split_source"].value_counts(dropna=False).astype(int).to_dict() if "season_split_source" in selected_df_future.columns else {},
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
    print(json.dumps(split_summary, indent=2))
    (outputs_path / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True)
    )

    print("Loading virus-only sequence pool...")
    sequence_pool_df = load_sequence_pool(sequence_pool_fasta, sequence_pool_metadata)
    sequence_pool_df = sequence_pool_df.reset_index(drop=True)

    if args.split_mode == "season":
        sequence_pool_df, excluded_sequence_pool_df = filter_sequence_pool_by_cutoff_date(
            sequence_pool_df,
            cutoff_season=args.cutoff_season,
            date_col="date",
        )
        excluded_pool_path = outputs_path / "virus_only_sequence_pool_excluded_by_cutoff.csv"
        excluded_sequence_pool_df.to_csv(excluded_pool_path, index=False)
        print(f"Saved virus-only rows excluded by cutoff: {excluded_pool_path}")

    print(f"Virus-only sequence pool size before sampling: {len(sequence_pool_df)}")
    sequence_pool_df = sample_virus_only_pool(
        sequence_pool_df,
        train_size=len(selected_df_train),
        k=args.virus_only_ratio_k,
        seed=split_seed,
    )
    print(f"Virus-only sequence pool size after sampling: {len(sequence_pool_df)}")
    sequence_pool_df.to_csv(outputs_path / "virus_only_sequence_pool_sampled.csv", index=False)

    print("Tokenizing sequences...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_max_length = args.max_length + 2
    print(
        f"Raw amino-acid length={args.max_length}; "
        f"ESM token_max_length={token_max_length} (<cls> + sequence + <eos>)"
    )
    dataset_train = make_paired_dataset(selected_df_train, tokenizer, token_max_length)
    dataset_val = make_paired_dataset(selected_df_val, tokenizer, token_max_length)
    dataset_test = make_paired_dataset(selected_df_test, tokenizer, token_max_length)
    dataset_future = (
        make_paired_dataset(selected_df_future, tokenizer, token_max_length)
        if has_future
        else None
    )
    dataset_virus_only = make_virus_only_dataset(sequence_pool_df, tokenizer, token_max_length)
    dataset_train_combined = ConcatDataset([dataset_train, dataset_virus_only])

    print("Fitting one-hot encoders...")
    ohe_virus = make_one_hot_encoder(selected_df_train["virus_category"])
    ohe_ref = make_one_hot_encoder(selected_df_train["reference_category"])
    ohe_date = make_one_hot_encoder(selected_df_train["date_category"])
    ohe_vp = make_one_hot_encoder(selected_df_train["virus_passage_category"])
    ohe_rp = make_one_hot_encoder(selected_df_train["reference_passage_category"])
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

    if has_future:
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
    lora_target_modules = parse_lora_target_modules(args.lora_target_modules)
    model = semanticESM(
        esm_config,
        args.model,
        effects_len=effects_len,
        virus_effects_len=virus_effects_len,
        embed_scale_factor=embed_scale_factor,
        CSE_W=args.CSE_w,
        SEMANTIC_W=args.semantic_w,
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
        use_systematic_error=args.use_systematic_error,
        freeze_esm=args.freeze_esm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_bias=args.lora_bias,
    )
    model.set_encoders(ohe_virus, ohe_ref, ohe_vp, ohe_rp)
    print(
        "Systematic-error correction: "
        + ("enabled" if args.use_systematic_error else "disabled (ablation)")
    )

    optimizer = build_plant_optimizer(
        model,
        learning_rate=args.learning_rate,
        encoder_learning_rate=args.encoder_learning_rate,
        regressor_learning_rate=args.regressor_learning_rate,
        auxiliary_learning_rate=args.auxiliary_learning_rate,
        weight_decay=args.weight_decay,
        regressor_weight_decay=args.reg_weight_decay,
    )
    resolved_encoder_lr = (
        args.learning_rate
        if args.encoder_learning_rate is None
        else args.encoder_learning_rate
    )
    resolved_regressor_lr = (
        args.learning_rate
        if args.regressor_learning_rate is None
        else args.regressor_learning_rate
    )
    resolved_auxiliary_lr = (
        resolved_regressor_lr
        if args.auxiliary_learning_rate is None
        else args.auxiliary_learning_rate
    )
    print(
        "Optimizer learning rates: "
        f"encoder={resolved_encoder_lr:.6g}, "
        f"regressor={resolved_regressor_lr:.6g}, "
        f"auxiliary={resolved_auxiliary_lr:.6g}"
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
        metric_for_best_model="eval_selection_score_censor_cap",
        greater_is_better=True,
        report_to="none",
    )

    trainer_callbacks = []
    if args.early_stopping_patience > 0:
        trainer_callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
        print(
            "Early stopping enabled: "
            f"patience={args.early_stopping_patience}, "
            f"threshold={args.early_stopping_threshold:.6g}"
        )
    else:
        print("Early stopping disabled (--early-stopping-patience=0).")

    trainer = BalancedCombinationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_combined,
        eval_dataset=dataset_val,
        num_samples_per_combination=args.num_samples_per_combination,
        random_seed=args.random_seed,
        optimizers=(optimizer, None),
        compute_metrics=partial(
            compute_metrics_for_trainer,
            selection_mae_lambda=args.selection_mae_lambda,
        ),
        callbacks=trainer_callbacks,
    )

    print("Starting training...")
    if args.checkpoint:
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        trainer.train()
    print("Training completed.")

    # Dump the complete evaluation history straight from the in-memory trainer state.
    #
    # Do not rely on results/checkpoint-*/trainer_state.json for this. With
    # save_total_limit=1 and load_best_model_at_end=True the surviving checkpoint can
    # be the BEST one, whose log_history stops at the step it was written -- which is
    # by construction the best-so-far step. Reading it back yields a curve that always
    # ends on its own maximum, which silently breaks any early-stopping replay or
    # peak-step analysis. trainer.state.log_history holds every evaluation regardless
    # of checkpoint rotation.
    log_history_path = outputs_path / "log_history.json"
    eval_curve = [
        {"step": int(entry["step"]), "value": float(entry["eval_selection_score_censor_cap"])}
        for entry in trainer.state.log_history
        if "eval_selection_score_censor_cap" in entry and "step" in entry
    ]
    log_history_path.write_text(
        json.dumps(
            {
                "metric": "eval_selection_score_censor_cap",
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "global_step": trainer.state.global_step,
                "eval_curve": eval_curve,
                "log_history": trainer.state.log_history,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Saved {len(eval_curve)} evaluations to {log_history_path}")

    print("Saving final model and tokenizer...")
    final_model_dir = outputs_path / "model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    run_config = vars(args).copy()
    run_config.update(
        {
            "bf16": bf16,
            "fp16": fp16,
            "token_max_length": token_max_length,
            "embed_scale_factor": embed_scale_factor,
            "resolved_learning_rates": {
                "encoder": resolved_encoder_lr,
                "regressor": resolved_regressor_lr,
                "auxiliary": resolved_auxiliary_lr,
            },
            "validation_selection_metric": (
                "pearson_censor_cap + spearman_censor_cap - "
                f"{args.selection_mae_lambda} * mae_censor_cap"
            ),
            "model_dir": str(final_model_dir),
            "plant_model_config": model.get_plant_init_config(),
            "category_mappings_file": str(category_mappings_path),
            "category_mapping_columns": list(CATEGORY_MAPPING_COLUMNS),
            "category_mapping_counts": {
                col: len(category_mappings[col]) for col in CATEGORY_MAPPING_COLUMNS
            },
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

    heldout_summary = {
        "split_mode": args.split_mode,
        "test": {
            "pearson_censor_cap": test_metrics["pearson_censor_cap"],
            "spearman_censor_cap": test_metrics["spearman_censor_cap"],
            "mae_censor_cap": test_metrics["mae_censor_cap"],
            "rmse_censor_cap": test_metrics["rmse_censor_cap"],
        },
    }

    if has_future:
        selected_df_future, future_metrics = predict_and_save_split(
            trainer,
            dataset_future,
            selected_df_future,
            outputs_path,
            split_name="future",
        )
        _future_season_metrics_df, future_season_metrics = write_future_metrics_by_season(
            selected_df_future,
            outputs_path,
            season_col="season_normalized",
        )
        heldout_summary["future"] = {
            "pearson_censor_cap": future_metrics["pearson_censor_cap"],
            "spearman_censor_cap": future_metrics["spearman_censor_cap"],
            "mae_censor_cap": future_metrics["mae_censor_cap"],
            "rmse_censor_cap": future_metrics["rmse_censor_cap"],
        }
        heldout_summary["future_by_season"] = {
            season: {
                "n": values.get("n"),
                "pearson_censor_cap": values.get("pearson_censor_cap"),
                "spearman_censor_cap": values.get("spearman_censor_cap"),
                "mae_censor_cap": values.get("mae_censor_cap"),
                "rmse_censor_cap": values.get("rmse_censor_cap"),
            }
            for season, values in future_season_metrics.items()
        }
    else:
        print("Full split mode selected: skipping future prediction and per-season future metrics.")
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
        encodes_gisaid = tokenize_sequences(gisaid_df["seq"].tolist(), tokenizer, token_max_length)
        dataset_gisaid = TextDataset(encodes_gisaid)
        dataloader_gisaid = DataLoader(dataset_gisaid, batch_size=args.embedding_batch_size, shuffle=False)
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

"""Inference helpers for PLANT.

This module supports two common inference workflows:

1. Sequence embedding / antigenic-map coordinates
   - input: HA sequence(s)
   - output: PLANT latent coordinates, e.g. z1/z2/z3
   - metadata/category mappings are not required

2. Pairwise antigenic-distance prediction
   - input: virus/reference HA sequence pairs
   - output: predicted_dist and predicted_dist_cartography
   - if ``use_systematic_error=True``, original metadata strings are converted to
     integer ``*_category`` values using ``category_mappings.json`` before the
     saved OneHotEncoder objects are applied inside the model.

The training code creates integer category columns first, then fits the
OneHotEncoder objects on those integer columns.  Therefore, for systematic-error
inference with new data, the string -> integer mapping must be saved and loaded
separately from the OneHotEncoder joblib files.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as safe_load
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmConfig

from .data import TextDataset, tokenize_sequences
from .model import semanticESM, set_encoders


CATEGORY_MAPPING_COLUMNS: tuple[str, ...] = (
    "date",
    "virus",
    "reference",
    "virus_passage",
    "reference_passage",
)

SYSTEMATIC_ERROR_COLUMNS: tuple[str, ...] = (
    "virus",
    "reference",
    "virus_passage",
    "reference_passage",
)

ENCODER_FILENAMES: dict[str, str] = {
    "virus": "virus_encoder.joblib",
    "reference": "ref_encoder.joblib",
    "date": "date_encoder.joblib",
    "virus_passage": "vp_encoder.joblib",
    "reference_passage": "rp_encoder.joblib",
}


@dataclass
class PlantInferenceArtifacts:
    """Objects loaded from a saved PLANT training run."""

    model: semanticESM
    tokenizer: Any
    model_dir: Path
    artifacts_dir: Path
    device: torch.device
    training_config: dict[str, Any]
    plant_model_config: dict[str, Any]
    encoders: dict[str, Any]
    category_mappings: Optional[dict[str, Any]]
    max_length: int
    use_bf16: bool
    use_fp16: bool


def _mixed_precision_dtype(
    device: torch.device,
    *,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> Optional[torch.dtype]:
    """Choose the inference autocast dtype.

    The model weights remain in their loaded dtype.  Mixed precision is applied
    only through ``torch.autocast`` during forward passes.
    """
    if device.type != "cuda":
        return None
    if use_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return None


def _autocast_context(
    device: torch.device,
    *,
    use_bf16: bool = True,
    use_fp16: bool = False,
):
    mixed_dtype = _mixed_precision_dtype(device, use_bf16=use_bf16, use_fp16=use_fp16)
    if mixed_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=mixed_dtype)
    return nullcontext()


def _json_load_if_exists(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_checkpoint_path(
    model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or Hugging Face Hub repo id to a local path."""
    path = Path(str(model_name_or_path)).expanduser()
    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - only needed for Hub loading
        raise FileNotFoundError(
            f"Local path does not exist: {model_name_or_path!r}. "
            "Install huggingface_hub to load a Hugging Face repo id."
        ) from exc

    return Path(
        snapshot_download(str(model_name_or_path), local_files_only=local_files_only)
    ).resolve()


def resolve_model_and_artifacts_dirs(
    model_name_or_path: str | Path,
    *,
    local_files_only: bool = False,
) -> tuple[Path, Path]:
    """Return ``(model_dir, artifacts_dir)`` for a saved PLANT run.

    Supported layouts:

    A. Training-run root::

        full_model/
          model/model.safetensors
          training_config.json
          category_mappings.json
          *_encoder.joblib

    B. Model directory only::

        model/
          model.safetensors
          plant_model_config.json
          config.json

    C. Hugging Face snapshot with either layout A or B.
    """
    root = _resolve_checkpoint_path(model_name_or_path, local_files_only=local_files_only)

    if (root / "model" / "model.safetensors").exists() or list(
        (root / "model").glob("model-*-of-*.safetensors")
    ):
        return root / "model", root

    if (root / "model.safetensors").exists() or list(root.glob("model-*-of-*.safetensors")):
        # If only the model directory was supplied, artifacts may either be in the
        # same directory or in the parent directory.
        parent_has_artifacts = any((root.parent / name).exists() for name in ENCODER_FILENAMES.values())
        parent_has_artifacts = parent_has_artifacts or (root.parent / "training_config.json").exists()
        artifacts_dir = root.parent if parent_has_artifacts else root
        return root, artifacts_dir

    raise FileNotFoundError(
        "Could not find model.safetensors or sharded model-*-of-*.safetensors under "
        f"{root}. Provide either the training-run root or the model/ directory."
    )


def _find_training_config(model_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    for candidate in (
        artifacts_dir / "training_config.json",
        model_dir / "training_config.json",
        model_dir.parent / "training_config.json",
    ):
        cfg = _json_load_if_exists(candidate)
        if cfg is not None:
            return cfg
    return {}


def load_category_mappings(
    artifacts_dir: str | Path,
    *,
    training_config: Optional[Mapping[str, Any]] = None,
    category_mappings_path: str | Path | None = None,
    required: bool = False,
) -> Optional[dict[str, Any]]:
    """Load ``category_mappings.json`` used for metadata string -> category ids.

    Search order:
    1. Explicit ``category_mappings_path``
    2. ``training_config['category_mappings_file']`` if present and reachable
    3. ``artifacts_dir / 'category_mappings.json'``
    """
    artifacts_dir = Path(artifacts_dir)
    candidates: list[Path] = []

    if category_mappings_path is not None:
        candidates.append(Path(category_mappings_path))

    if training_config is not None:
        configured = training_config.get("category_mappings_file")
        if configured:
            candidates.append(Path(str(configured)))
            # If the config stores an absolute path from another machine/runtime,
            # also try the same basename in artifacts_dir.
            candidates.append(artifacts_dir / Path(str(configured)).name)

    candidates.append(artifacts_dir / "category_mappings.json")

    seen: set[Path] = set()
    for path in candidates:
        path = path.expanduser()
        if path in seen:
            continue
        seen.add(path)
        cfg = _json_load_if_exists(path)
        if cfg is not None:
            return cfg

    if required:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"category_mappings.json was not found. Tried: {tried}")
    return None


def load_systematic_error_encoders(
    artifacts_dir: str | Path,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Load saved OneHotEncoder objects and register them in ``plant.model``."""
    artifacts_dir = Path(artifacts_dir)
    encoders: dict[str, Any] = {}
    missing: list[str] = []

    for key, filename in ENCODER_FILENAMES.items():
        path = artifacts_dir / filename
        if path.exists():
            encoders[key] = joblib.load(path)
        else:
            encoders[key] = None
            missing.append(filename)

    if required and missing:
        raise FileNotFoundError(
            "Missing required systematic-error encoder files in "
            f"{artifacts_dir}: {missing}"
        )

    set_encoders(
        encoders.get("virus"),
        encoders.get("reference"),
        encoders.get("virus_passage"),
        encoders.get("reference_passage"),
    )
    return encoders


def _load_safetensors_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    files: list[Path] = []
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        files.append(single_file)
    files.extend(sorted(model_dir.glob("model-*-of-*.safetensors")))

    if not files:
        raise FileNotFoundError(
            f"No model.safetensors or sharded model-*-of-*.safetensors files found in {model_dir}."
        )

    state_dict: dict[str, torch.Tensor] = {}
    for file in files:
        state_dict.update(safe_load(str(file), device="cpu"))
    return state_dict


def _remap_layernorm_gamma_beta(
    state_dict: Mapping[str, torch.Tensor],
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], int]:
    """Handle older ESM LayerNorm gamma/beta key names."""
    remapped: dict[str, torch.Tensor] = {}
    n_renamed = 0

    for key, value in state_dict.items():
        new_key = key

        if ".LayerNorm.gamma" in new_key:
            candidate = new_key.replace(".LayerNorm.gamma", ".LayerNorm.weight")
            if candidate in target_keys:
                new_key = candidate

        if ".LayerNorm.beta" in new_key:
            candidate = new_key.replace(".LayerNorm.beta", ".LayerNorm.bias")
            if candidate in target_keys:
                new_key = candidate

        if new_key != key:
            n_renamed += 1
        remapped[new_key] = value

    return remapped, n_renamed


def load_plant_inference(
    model_name_or_path: str | Path,
    *,
    device: str | torch.device | None = None,
    use_bf16: Optional[bool] = None,
    use_fp16: Optional[bool] = None,
    category_mappings_path: str | Path | None = None,
    require_category_mappings: bool = False,
    require_encoders: bool = False,
    strict: bool = True,
    local_files_only: bool = False,
) -> PlantInferenceArtifacts:
    """Load a PLANT model plus tokenizer, encoders, and category mappings.

    Parameters
    ----------
    model_name_or_path:
        Local training-run root, local ``model/`` directory, or Hugging Face Hub
        repo id.  If a training-run root is supplied, files such as
        ``training_config.json``, ``category_mappings.json``, and encoder joblibs
        are loaded from that root.
    use_bf16/use_fp16:
        If omitted, values are read from ``training_config.json`` when available.
    require_category_mappings:
        Set ``True`` when you plan to run systematic-error prediction from
        metadata strings and want loading to fail loudly if mappings are missing.
    require_encoders:
        Set ``True`` when systematic-error prediction is required.
    strict:
        If ``True``, raise if checkpoint keys do not exactly match after the
        LayerNorm gamma/beta compatibility remapping.
    """
    model_dir, artifacts_dir = resolve_model_and_artifacts_dirs(
        model_name_or_path, local_files_only=local_files_only
    )
    training_config = _find_training_config(model_dir, artifacts_dir)

    if use_bf16 is None:
        use_bf16 = bool(training_config.get("bf16", True))
    if use_fp16 is None:
        use_fp16 = bool(training_config.get("fp16", False))

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    plant_config_path = model_dir / "plant_model_config.json"
    if not plant_config_path.exists():
        raise FileNotFoundError(f"Missing plant_model_config.json in {model_dir}")
    plant_model_config = json.loads(plant_config_path.read_text(encoding="utf-8"))

    encoders = load_systematic_error_encoders(artifacts_dir, required=require_encoders)
    category_mappings = load_category_mappings(
        artifacts_dir,
        training_config=training_config,
        category_mappings_path=category_mappings_path,
        required=require_category_mappings,
    )

    esm_config = EsmConfig.from_pretrained(model_dir)
    model = semanticESM(esm_config, **plant_model_config)

    raw_state_dict = _load_safetensors_state_dict(model_dir)
    state_dict, n_renamed = _remap_layernorm_gamma_beta(
        raw_state_dict, set(model.state_dict().keys())
    )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys or unexpected_keys:
        message = (
            "[PLANT inference] Checkpoint keys do not match the reconstructed model. "
            f"Renamed LayerNorm keys: {n_renamed}; "
            f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    model.to(device_obj)
    model.eval()

    max_length = int(training_config.get("max_length", tokenizer.model_max_length))
    if max_length is None or max_length <= 0 or max_length > 100000:
        max_length = int(getattr(model.config, "max_position_embeddings", 329) or 329)

    return PlantInferenceArtifacts(
        model=model,
        tokenizer=tokenizer,
        model_dir=model_dir,
        artifacts_dir=artifacts_dir,
        device=device_obj,
        training_config=dict(training_config),
        plant_model_config=dict(plant_model_config),
        encoders=encoders,
        category_mappings=category_mappings,
        max_length=max_length,
        use_bf16=bool(use_bf16),
        use_fp16=bool(use_fp16),
    )


def _unknown_category_from_mappings(
    category_mappings: Optional[Mapping[str, Any]],
    default: int = -1,
) -> int:
    if not category_mappings:
        return default
    metadata = category_mappings.get("__metadata__", {})
    try:
        return int(metadata.get("unknown_category", default))
    except Exception:
        return default


def _encode_values_with_mapping(
    values: Sequence[Any],
    mapping: Mapping[str, Any],
    *,
    unknown_category: int = -1,
    strict: bool = False,
    column_name: str = "metadata",
) -> tuple[list[int], list[str]]:
    encoded: list[int] = []
    unknown_values: list[str] = []

    for value in values:
        if pd.isna(value):
            encoded.append(unknown_category)
            unknown_values.append("<NA>")
            continue

        key = str(value)
        if key in mapping:
            encoded.append(int(mapping[key]))
        else:
            encoded.append(unknown_category)
            unknown_values.append(key)

    if strict and unknown_values:
        examples = sorted(set(unknown_values))[:20]
        raise KeyError(
            f"Unknown values in {column_name!r}: {examples}. "
            "Add these values to the training data/mapping or use strict=False "
            "to encode them as the unknown all-zero OneHotEncoder category."
        )

    return encoded, unknown_values


def apply_category_mappings(
    df: pd.DataFrame,
    category_mappings: Mapping[str, Any],
    *,
    columns: Sequence[str] = CATEGORY_MAPPING_COLUMNS,
    unknown_category: Optional[int] = None,
    strict: bool = False,
    keep_unknown_report: bool = True,
) -> pd.DataFrame:
    """Add ``*_category`` columns from original metadata strings.

    Unknown strings are encoded as ``unknown_category`` (default: value stored in
    ``category_mappings['__metadata__']['unknown_category']`` or ``-1``).  Because
    the training OneHotEncoder uses ``handle_unknown='ignore'``, this unknown id
    becomes an all-zero one-hot vector.
    """
    out = df.copy()
    if unknown_category is None:
        unknown_category = _unknown_category_from_mappings(category_mappings)

    unknown_report: dict[str, list[str]] = {}

    for col in columns:
        category_col = f"{col}_category"

        if category_col in out.columns:
            out[category_col] = out[category_col].fillna(unknown_category).astype(int)
            continue

        if col not in out.columns:
            if strict:
                raise KeyError(
                    f"Missing both {col!r} and {category_col!r}; cannot encode category."
                )
            out[category_col] = int(unknown_category)
            unknown_report[col] = ["<MISSING_COLUMN>"]
            continue

        if col not in category_mappings:
            if strict:
                raise KeyError(f"Column {col!r} is not present in category_mappings.json")
            out[category_col] = int(unknown_category)
            unknown_report[col] = ["<MISSING_MAPPING>"]
            continue

        encoded, unknown_values = _encode_values_with_mapping(
            out[col].tolist(),
            category_mappings[col],
            unknown_category=unknown_category,
            strict=strict,
            column_name=col,
        )
        out[category_col] = encoded
        if unknown_values:
            unknown_report[col] = sorted(set(unknown_values))[:50]

    if keep_unknown_report:
        out.attrs["category_unknown_report"] = unknown_report
    return out


def prepare_pairwise_metadata_categories(
    df: pd.DataFrame,
    *,
    category_mappings: Optional[Mapping[str, Any]] = None,
    use_systematic_error: bool = False,
    strict_metadata: bool = False,
    unknown_category: int = -1,
) -> pd.DataFrame:
    """Prepare category columns for pairwise distance prediction.

    If ``use_systematic_error=False``, all metadata categories are forced to the
    unknown id so the saved OneHotEncoder objects produce all-zero vectors and the
    output ``predicted_dist`` equals the cartography-style distance up to numeric
    precision.
    """
    out = df.copy()

    if not use_systematic_error:
        for col in CATEGORY_MAPPING_COLUMNS:
            out[f"{col}_category"] = int(unknown_category)
        out.attrs["category_unknown_report"] = {}
        return out

    if category_mappings is None:
        raise ValueError(
            "category_mappings.json is required when use_systematic_error=True. "
            "Load with load_plant_inference(..., require_category_mappings=True) "
            "or pass category_mappings explicitly."
        )

    out = apply_category_mappings(
        out,
        category_mappings,
        columns=CATEGORY_MAPPING_COLUMNS,
        unknown_category=unknown_category,
        strict=strict_metadata,
    )

    if strict_metadata:
        for col in SYSTEMATIC_ERROR_COLUMNS:
            if f"{col}_category" not in out.columns:
                raise KeyError(f"Missing required category column: {col}_category")

    return out


@torch.no_grad()
def embed_sequences(
    model,
    dataloader,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> np.ndarray:
    """Return PLANT latent coordinates for virus sequences."""
    model.eval()
    device = next(model.parameters()).device
    outs = []
    with _autocast_context(device, use_bf16=use_bf16, use_fp16=use_fp16):
        for batch in dataloader:
            outputs = model(
                input_ids_virus=batch["input_ids_virus"].to(device),
                attention_mask_virus=batch["attention_mask_virus"].to(device),
            )
            outs.append(outputs.hidden_state_virus.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def embed_sequences_dataframe(
    artifacts: PlantInferenceArtifacts,
    df: pd.DataFrame,
    *,
    seq_col: str = "seq",
    id_col: Optional[str] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Embed sequences from a dataframe and append latent coordinate columns."""
    if seq_col not in df.columns:
        raise KeyError(f"Missing sequence column: {seq_col!r}")

    encodes = tokenize_sequences(
        df[seq_col].astype(str).tolist(),
        artifacts.tokenizer,
        artifacts.max_length,
    )
    dataset = TextDataset(encodes, always_include_reference=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    coords = embed_sequences(
        artifacts.model,
        dataloader,
        use_bf16=artifacts.use_bf16,
        use_fp16=artifacts.use_fp16,
    )

    if id_col is not None and id_col in df.columns:
        out = df[[id_col, seq_col]].copy()
    else:
        out = df.copy()

    for idx in range(coords.shape[1]):
        out[f"z{idx + 1}"] = coords[:, idx]
    return out


@torch.no_grad()
def predict_pairwise_distances(
    artifacts: PlantInferenceArtifacts,
    df: pd.DataFrame,
    *,
    virus_seq_col: str = "virus_seq",
    reference_seq_col: str = "reference_seq",
    batch_size: int = 32,
    use_systematic_error: bool = False,
    strict_metadata: bool = False,
    unknown_category: Optional[int] = None,
    return_category_columns: bool = True,
) -> pd.DataFrame:
    """Predict antigenic distances for virus/reference sequence pairs.

    Parameters
    ----------
    use_systematic_error:
        If ``False`` (default), metadata categories are forced to the unknown id,
        making the OneHotEncoder outputs all-zero.  This is safest when metadata
        is unavailable and returns the sequence-coordinate/cartography distance.

        If ``True``, metadata strings such as ``virus_passage`` are converted to
        integer category ids using ``category_mappings.json`` and the model's
        systematic-error correction terms are applied.
    strict_metadata:
        If ``True``, unknown metadata strings raise an error.  If ``False``, they
        are encoded as the unknown all-zero one-hot category.
    """
    if virus_seq_col not in df.columns:
        raise KeyError(f"Missing virus sequence column: {virus_seq_col!r}")
    if reference_seq_col not in df.columns:
        raise KeyError(f"Missing reference sequence column: {reference_seq_col!r}")

    if unknown_category is None:
        unknown_category = _unknown_category_from_mappings(artifacts.category_mappings)

    work_df = prepare_pairwise_metadata_categories(
        df,
        category_mappings=artifacts.category_mappings,
        use_systematic_error=use_systematic_error,
        strict_metadata=strict_metadata,
        unknown_category=unknown_category,
    )

    # If systematic error is requested, all encoders used by the model should be loaded.
    if use_systematic_error:
        required_encoder_keys = ("virus", "reference", "virus_passage", "reference_passage")
        missing = [key for key in required_encoder_keys if artifacts.encoders.get(key) is None]
        if missing:
            raise FileNotFoundError(
                "Systematic-error prediction requires saved OneHotEncoder joblibs. "
                f"Missing encoders: {missing}"
            )

    encodes_virus = tokenize_sequences(
        work_df[virus_seq_col].astype(str).tolist(),
        artifacts.tokenizer,
        artifacts.max_length,
    )
    encodes_reference = tokenize_sequences(
        work_df[reference_seq_col].astype(str).tolist(),
        artifacts.tokenizer,
        artifacts.max_length,
    )

    n = len(work_df)
    dataset = TextDataset(
        encodes_virus,
        encodes_reference,
        labels=[0.0] * n,  # dummy non-missing labels are needed to obtain pairwise logits
        censors=[0.0] * n,
        virus=work_df["virus_category"].astype(int).tolist(),
        reference=work_df["reference_category"].astype(int).tolist(),
        dates=work_df["date_category"].astype(int).tolist(),
        virus_passage=work_df["virus_passage_category"].astype(int).tolist(),
        reference_passage=work_df["reference_passage_category"].astype(int).tolist(),
        weight=[1.0] * n,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = artifacts.model
    model.eval()
    device = artifacts.device
    logits_parts: list[np.ndarray] = []

    with _autocast_context(
        device,
        use_bf16=artifacts.use_bf16,
        use_fp16=artifacts.use_fp16,
    ):
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            outputs = model(**batch)
            if outputs.logits is None:
                raise RuntimeError("Model returned logits=None during pairwise prediction.")
            logits_parts.append(outputs.logits[:, :2].float().cpu().numpy())

    logits = np.concatenate(logits_parts, axis=0)

    out = df.copy().reset_index(drop=True)
    out["predicted_dist"] = logits[:, 0]
    out["predicted_dist_cartography"] = logits[:, 1]
    out["used_systematic_error"] = bool(use_systematic_error)

    if return_category_columns:
        for col in CATEGORY_MAPPING_COLUMNS:
            out[f"{col}_category"] = work_df[f"{col}_category"].astype(int).to_numpy()
        out.attrs["category_unknown_report"] = work_df.attrs.get("category_unknown_report", {})

    return out


def predict_pairwise_distances_from_sequences(
    artifacts: PlantInferenceArtifacts,
    virus_sequences: Sequence[str],
    reference_sequences: Sequence[str],
    *,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Convenience wrapper for metadata-free cartography distance prediction."""
    if len(virus_sequences) != len(reference_sequences):
        raise ValueError(
            "virus_sequences and reference_sequences must have the same length: "
            f"{len(virus_sequences)} != {len(reference_sequences)}"
        )
    df = pd.DataFrame(
        {
            "virus_seq": list(virus_sequences),
            "reference_seq": list(reference_sequences),
        }
    )
    return predict_pairwise_distances(
        artifacts,
        df,
        batch_size=batch_size,
        use_systematic_error=False,
    )


__all__ = [
    "CATEGORY_MAPPING_COLUMNS",
    "SYSTEMATIC_ERROR_COLUMNS",
    "PlantInferenceArtifacts",
    "resolve_model_and_artifacts_dirs",
    "load_category_mappings",
    "load_systematic_error_encoders",
    "load_plant_inference",
    "apply_category_mappings",
    "prepare_pairwise_metadata_categories",
    "embed_sequences",
    "embed_sequences_dataframe",
    "predict_pairwise_distances",
    "predict_pairwise_distances_from_sequences",
]

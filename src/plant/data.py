"""Dataset utilities for PLANT.

This module supports both inference-only use cases and mixed supervised / semantic
training.  The key design choice is that ``TextDataset`` can always emit the
reference-side keys, even when the example is virus-only.  This makes it safe to
combine paired HI/antigenic-distance examples and virus-only examples in a
``ConcatDataset``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


MISSING_LABEL_VALUE = -10.0


def _as_list(x: Optional[Sequence[Any]], n: int, default: Any) -> list[Any]:
    if x is None:
        return [default] * n
    if len(x) != n:
        raise ValueError(f"Expected length {n}, got {len(x)}")
    return list(x)


class TextDataset(Dataset):
    """PLANT dataset for paired or virus-only examples.

    Parameters
    ----------
    encodes_virus:
        Tokenized virus sequences returned by a Hugging Face tokenizer.
    encodes_reference:
        Tokenized reference sequences.  When omitted, reference tensors are
        filled with zeros by default so that mixed ``ConcatDataset`` objects can
        be collated safely.
    labels:
        Antigenic distances. Missing labels are represented by ``-10.0`` and
        exclude the row from supervised antigenic-distance loss. The sequence
        remains part of the unified semantic/CSE/LG regularization set.
    always_include_reference:
        If ``True`` (default), ``input_ids_reference`` and
        ``attention_mask_reference`` are included even for virus-only examples.
        Keep this as ``True`` for training with a mixture of paired and
        virus-only datasets.  It can be set to ``False`` for very small
        inference-only datasets if desired.
    """

    def __init__(
        self,
        encodes_virus: dict[str, torch.Tensor],
        encodes_reference: Optional[dict[str, torch.Tensor]] = None,
        labels: Optional[Sequence[float]] = None,
        censors: Optional[Sequence[float]] = None,
        virus: Optional[Sequence[int]] = None,
        reference: Optional[Sequence[int]] = None,
        dates: Optional[Sequence[int]] = None,
        virus_passage: Optional[Sequence[int]] = None,
        reference_passage: Optional[Sequence[int]] = None,
        weight: Optional[Sequence[float]] = None,
        *,
        always_include_reference: bool = True,
        missing_label_value: float = MISSING_LABEL_VALUE,
    ) -> None:
        self.input_ids_virus = encodes_virus["input_ids"]
        self.attention_mask_virus = encodes_virus["attention_mask"]
        self.n = len(self.input_ids_virus)
        self.has_reference = encodes_reference is not None
        self.always_include_reference = always_include_reference
        self.missing_label_value = missing_label_value

        if self.has_reference:
            self.input_ids_reference = encodes_reference["input_ids"]
            self.attention_mask_reference = encodes_reference["attention_mask"]
            if len(self.input_ids_reference) != self.n:
                raise ValueError(
                    "encodes_reference['input_ids'] and encodes_virus['input_ids'] "
                    f"must have the same length: {len(self.input_ids_reference)} != {self.n}"
                )
            if len(self.attention_mask_reference) != self.n:
                raise ValueError(
                    "encodes_reference['attention_mask'] and encodes_virus['attention_mask'] "
                    f"must have the same length: {len(self.attention_mask_reference)} != {self.n}"
                )
        else:
            self.input_ids_reference = None
            self.attention_mask_reference = None

        self.labels = _as_list(labels, self.n, None)
        self.censors = _as_list(censors, self.n, 0.0)
        self._virus_provided = virus is not None
        self.virus = _as_list(virus, self.n, 0)
        self.reference = _as_list(reference, self.n, 0)
        self.dates = _as_list(dates, self.n, 0)
        self.virus_passage = _as_list(virus_passage, self.n, 0)
        self.reference_passage = _as_list(reference_passage, self.n, 0)
        self.weight = _as_list(weight, self.n, 1.0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        label = self.labels[idx]
        item = {
            "input_ids_virus": self.input_ids_virus[idx],
            "attention_mask_virus": self.attention_mask_virus[idx],
            "labels": torch.tensor(
                self.missing_label_value if label is None else label,
                dtype=torch.float,
            ),
            "censors": torch.tensor(self.censors[idx], dtype=torch.float),
            "virus": torch.tensor(self.virus[idx], dtype=torch.long),
            "reference": torch.tensor(self.reference[idx], dtype=torch.long),
            "dates": torch.tensor(self.dates[idx], dtype=torch.long),
            "virus_passage": torch.tensor(self.virus_passage[idx], dtype=torch.long),
            "reference_passage": torch.tensor(
                self.reference_passage[idx], dtype=torch.long
            ),
            "weight": torch.tensor(self.weight[idx], dtype=torch.float),
        }

        if self.has_reference:
            item["input_ids_reference"] = self.input_ids_reference[idx]
            item["attention_mask_reference"] = self.attention_mask_reference[idx]
        elif self.always_include_reference:
            item["input_ids_reference"] = torch.zeros_like(self.input_ids_virus[idx])
            item["attention_mask_reference"] = torch.zeros_like(
                self.attention_mask_virus[idx]
            )

        return item

    def __len__(self) -> int:
        return self.n

    def get_unique_combinations_indices(self) -> dict[tuple[Any, ...], list[int]]:
        """Return indices grouped by experimental combination.

        Paired examples are grouped by virus/reference/passage categories.  Virus-only
        examples are grouped by virus category so that the balanced sampler can draw
        a limited number of examples from each repeated sequence or strain.
        """
        unique: dict[tuple[Any, ...], list[int]] = {}
        for idx in range(self.n):
            label = self.labels[idx]
            has_label = label is not None and float(label) != self.missing_label_value
            if has_label:
                key = (
                    self.virus[idx],
                    self.reference[idx],
                    self.virus_passage[idx],
                    self.reference_passage[idx],
                )
            else:
                # If no virus/strain category is provided for sequence-only data,
                # treat each row as its own group.  This avoids sampling only one
                # sequence from the entire unlabeled pool.
                key = (self.virus[idx],) if self._virus_provided else (idx,)
            unique.setdefault(key, []).append(idx)
        return unique


def tokenize_sequences(
    seq: Sequence[str] | Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize amino-acid sequences for PLANT/ESM."""
    return tokenizer(
        list(seq),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

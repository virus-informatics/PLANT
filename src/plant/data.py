# src/plant/data.py
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# define dataset class
class TextDataset(Dataset):
    def __init__(
        self,
        encodes_virus,
        encodes_reference=None,
        labels=None,
        censors=None,
        virus=None,
        reference=None,
        dates=None,
        virus_passage=None,
        reference_passage=None,
        weight=None,
    ):
        self.input_ids_virus = encodes_virus["input_ids"]
        self.attention_mask_virus = encodes_virus["attention_mask"]
        n = len(self.input_ids_virus)

        self.has_reference = encodes_reference is not None
        if self.has_reference:
            self.input_ids_reference = encodes_reference["input_ids"]
            self.attention_mask_reference = encodes_reference["attention_mask"]
            assert len(self.input_ids_reference) == n, \
                f"len(input_ids_reference)={len(self.input_ids_reference)} != len(input_ids_virus)={n}"
            assert len(self.attention_mask_reference) == n, \
                f"len(attention_mask_reference)={len(self.attention_mask_reference)} != len(attention_mask_virus)={n}"
        else:
            self.input_ids_reference = None
            self.attention_mask_reference = None

        def fill(x, default):
            return x if x is not None else [default] * n

        self.labels            = fill(labels,            None)
        self.censors           = fill(censors,           None)
        self.virus             = fill(virus,             None)
        self.reference         = fill(reference,         None)
        self.dates             = fill(dates,             None)
        self.virus_passage     = fill(virus_passage,     None)
        self.reference_passage = fill(reference_passage, None)
        self.weight            = fill(weight,            1.0)

    def __getitem__(self, idx):
        item = {
            "input_ids_virus": self.input_ids_virus[idx],
            "attention_mask_virus": self.attention_mask_virus[idx],
            "labels":  torch.tensor(self.labels[idx]  if self.labels[idx]  is not None else -10.0, dtype=torch.float),
            "censors": torch.tensor(self.censors[idx] if self.censors[idx] is not None else 0.0,  dtype=torch.float),
            "virus":   torch.tensor(self.virus[idx]   if self.virus[idx]   is not None else 0,    dtype=torch.long),
            "reference": torch.tensor(self.reference[idx] if self.reference[idx] is not None else 0, dtype=torch.long),
            "dates":  torch.tensor(self.dates[idx]    if self.dates[idx]    is not None else 0,    dtype=torch.long),
            "virus_passage": torch.tensor(self.virus_passage[idx] if self.virus_passage[idx] is not None else 0, dtype=torch.long),
            "reference_passage": torch.tensor(self.reference_passage[idx] if self.reference_passage[idx] is not None else 0, dtype=torch.long),
            "weight": torch.tensor(self.weight[idx]  if self.weight[idx]  is not None else 1.0,  dtype=torch.float),
        }

        if self.has_reference:
            item["input_ids_reference"] = self.input_ids_reference[idx]
            item["attention_mask_reference"] = self.attention_mask_reference[idx]
        return item

    def __len__(self):
        return len(self.input_ids_virus)

    def get_unique_combinations_indices(self):
        unique = {}
        for idx in range(len(self.input_ids_virus)):
            if self.has_reference:
                key = (self.virus[idx], self.reference[idx], self.virus_passage[idx], self.reference_passage[idx])
            else:
                key = (self.virus[idx],)
            unique.setdefault(key, []).append(idx)
        return unique


def tokenize_sequences(seq, tokenizer: PreTrainedTokenizerBase, MAX_LENGTH: int):
    return tokenizer(
        seq, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_attention_mask=True, return_tensors="pt",
    )



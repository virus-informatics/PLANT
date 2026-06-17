"""Training utilities for PLANT."""

from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, RandomSampler, Subset
from transformers import AutoModel, EsmConfig, PreTrainedModel, Trainer
from transformers.utils import ModelOutput


class BalancedCombinationTrainer(Trainer):
    """Trainer that samples at most N examples per experimental combination.

    The original training script balanced repeated HI/antigenic-distance
    measurements by drawing one row per unique combination at each epoch.  This
    implementation keeps that behavior and correctly handles ``ConcatDataset``
    offsets when paired and virus-only datasets are mixed.
    """

    def __init__(
        self,
        *args,
        num_samples_per_combination: int = 1,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_samples_per_combination = num_samples_per_combination
        self.random_seed = 0 if random_seed is None else random_seed
        self._epoch_for_sampling = 0

    def _sample_dataset_indices(self, dataset, offset: int, generator: torch.Generator):
        if hasattr(dataset, "get_unique_combinations_indices"):
            unique_combinations = dataset.get_unique_combinations_indices()
            sampled_indices: list[int] = []
            for indices in unique_combinations.values():
                if len(indices) <= self.num_samples_per_combination:
                    sampled_indices.extend(offset + i for i in indices)
                else:
                    perm = torch.randperm(len(indices), generator=generator).tolist()
                    sampled_indices.extend(
                        offset + indices[i]
                        for i in perm[: self.num_samples_per_combination]
                    )
            return sampled_indices
        return list(range(offset, offset + len(dataset)))

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        generator = torch.Generator()
        generator.manual_seed(self.random_seed + self._epoch_for_sampling)
        random.seed(self.random_seed + self._epoch_for_sampling)
        torch.manual_seed(self.random_seed + self._epoch_for_sampling)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed + self._epoch_for_sampling)
        self._epoch_for_sampling += 1

        if isinstance(train_dataset, ConcatDataset):
            subsampled_indices: list[int] = []
            offset = 0
            for dataset in train_dataset.datasets:
                subsampled_indices.extend(
                    self._sample_dataset_indices(dataset, offset, generator)
                )
                offset += len(dataset)
        else:
            subsampled_indices = self._sample_dataset_indices(train_dataset, 0, generator)

        subsampled_dataset = Subset(train_dataset, subsampled_indices)
        sampler = RandomSampler(subsampled_dataset, generator=generator)
        batch_sampler = BatchSampler(
            sampler,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        return DataLoader(
            subsampled_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def build_plant_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    regressor_weight_decay: float,
) -> AdamW:
    """Create an AdamW optimizer with a separate regressor weight decay."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    regressor_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "regressor" in name:
            regressor_params.append((name, param))
        else:
            other_params.append((name, param))

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in regressor_params],
            "weight_decay": regressor_weight_decay,
        },
        {
            "params": [
                p for n, p in other_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in other_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate)


class ESMEmbeddingDistanceModel(PreTrainedModel):
    """Frozen ESM mean+max embedding-distance helper."""

    config_class = EsmConfig

    def __init__(self, config: EsmConfig, esm_model_name: str) -> None:
        super().__init__(config)
        self.esm_model = AutoModel.from_pretrained(
            esm_model_name, add_pooling_layer=False
        )
        self.embedding_dim = self.esm_model.config.hidden_size
        self.eval()

    def forward(
        self,
        input_ids_virus: torch.Tensor,
        attention_mask_virus: torch.Tensor,
        input_ids_reference: torch.Tensor,
        attention_mask_reference: torch.Tensor,
        **kwargs,
    ) -> ModelOutput:
        del kwargs
        virus_encoder_out = self.esm_model(
            input_ids_virus, attention_mask=attention_mask_virus
        ).last_hidden_state
        reference_encoder_out = self.esm_model(
            input_ids_reference, attention_mask=attention_mask_reference
        ).last_hidden_state

        masked_sum_virus = (virus_encoder_out * attention_mask_virus.unsqueeze(-1)).sum(
            dim=1
        )
        mask_count_virus = attention_mask_virus.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled_virus = masked_sum_virus / mask_count_virus

        masked_sum_reference = (
            reference_encoder_out * attention_mask_reference.unsqueeze(-1)
        ).sum(dim=1)
        mask_count_reference = attention_mask_reference.sum(dim=1, keepdim=True).clamp(
            min=1
        )
        mean_pooled_reference = masked_sum_reference / mask_count_reference

        virus_mask = attention_mask_virus.unsqueeze(-1).bool()
        reference_mask = attention_mask_reference.unsqueeze(-1).bool()
        virus_encoder_out = virus_encoder_out.masked_fill(
            ~virus_mask, torch.finfo(virus_encoder_out.dtype).min
        )
        reference_encoder_out = reference_encoder_out.masked_fill(
            ~reference_mask, torch.finfo(reference_encoder_out.dtype).min
        )
        max_pooled_virus = torch.max(virus_encoder_out, dim=1)[0]
        max_pooled_reference = torch.max(reference_encoder_out, dim=1)[0]

        virus_embedding = torch.cat([mean_pooled_virus, max_pooled_virus], dim=-1)
        reference_embedding = torch.cat(
            [mean_pooled_reference, max_pooled_reference], dim=-1
        )
        distance = torch.norm(virus_embedding - reference_embedding, p=2, dim=1, keepdim=True)
        return ModelOutput(logits=distance)


@torch.no_grad()
def compute_embedding_distances(
    dataset,
    *,
    esm_model_name: str,
    batch_size: int = 128,
    device: Optional[torch.device | str] = None,
    use_fp16: bool = True,
) -> np.ndarray:
    """Compute frozen ESM embedding distances for paired examples."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = EsmConfig.from_pretrained(esm_model_name)
    model = ESMEmbeddingDistanceModel(config, esm_model_name).to(device)
    if use_fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    distances = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and device.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        for batch in dataloader:
            outputs = model(
                input_ids_virus=batch["input_ids_virus"].to(device),
                attention_mask_virus=batch["attention_mask_virus"].to(device),
                input_ids_reference=batch["input_ids_reference"].to(device),
                attention_mask_reference=batch["attention_mask_reference"].to(device),
            )
            distances.append(outputs.logits.float().cpu().numpy())
    return np.concatenate(distances, axis=0).reshape(-1)


def estimate_embed_scale_factor(
    dataset,
    *,
    esm_model_name: str,
    quantile: float = 0.99,
    batch_size: int = 128,
    device: Optional[torch.device | str] = None,
    use_fp16: bool = True,
) -> float:
    """Estimate the semantic-loss scale factor used in PLANT training."""
    distances = compute_embedding_distances(
        dataset,
        esm_model_name=esm_model_name,
        batch_size=batch_size,
        device=device,
        use_fp16=use_fp16,
    )
    return float(np.quantile(distances, quantile))

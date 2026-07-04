"""Training utilities for PLANT."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, Sampler
from transformers import AutoModel, EsmConfig, PreTrainedModel, Trainer
from transformers.utils import ModelOutput


def _mixed_precision_dtype(
    device: torch.device,
    *,
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> Optional[torch.dtype]:
    """Choose the mixed-precision dtype for CUDA helper inference.

    BF16 is preferred when requested and supported.  FP16 is kept as an
    explicit fallback for older GPUs.  CPU execution uses full precision.
    """
    if device.type != "cuda":
        return None
    if use_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return None


class BalancedCombinationSampler(Sampler[int]):
    """Sample at most N examples per experimental combination at each epoch.

    The sampled index list is rebuilt every time ``__iter__`` is called.  In the
    Hugging Face Trainer training loop, this corresponds to rebuilding the
    sampled replicate set at the start of each epoch.
    """

    def __init__(
        self,
        dataset,
        *,
        num_samples_per_combination: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if num_samples_per_combination < 1:
            raise ValueError("num_samples_per_combination must be >= 1")
        self.dataset = dataset
        self.num_samples_per_combination = num_samples_per_combination
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

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

    def _sample_indices_for_epoch(self, epoch: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        if isinstance(self.dataset, ConcatDataset):
            sampled_indices: list[int] = []
            offset = 0
            for dataset in self.dataset.datasets:
                sampled_indices.extend(
                    self._sample_dataset_indices(dataset, offset, generator)
                )
                offset += len(dataset)
        else:
            sampled_indices = self._sample_dataset_indices(
                self.dataset,
                offset=0,
                generator=generator,
            )

        if self.shuffle:
            perm = torch.randperm(len(sampled_indices), generator=generator).tolist()
            sampled_indices = [sampled_indices[i] for i in perm]

        return sampled_indices

    def __iter__(self):
        # DataLoader calls sampler.__iter__() whenever a new pass over the
        # dataloader starts.  With Trainer, that is effectively once per epoch.
        current_epoch = self.epoch
        sampled_indices = self._sample_indices_for_epoch(current_epoch)
        self.epoch = current_epoch + 1
        return iter(sampled_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch explicitly when a training loop/accelerator supports it."""
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))

    def __len__(self) -> int:
        if isinstance(self.dataset, ConcatDataset):
            total = 0
            for dataset in self.dataset.datasets:
                total += self._sampled_length(dataset)
            return total
        return self._sampled_length(self.dataset)

    def _sampled_length(self, dataset) -> int:
        if hasattr(dataset, "get_unique_combinations_indices"):
            unique_combinations = dataset.get_unique_combinations_indices()
            return sum(
                min(len(indices), self.num_samples_per_combination)
                for indices in unique_combinations.values()
            )
        return len(dataset)


class BalancedCombinationTrainer(Trainer):
    """Trainer that samples at most N examples per experimental combination.

    Paired HI/antigenic-distance examples are balanced by drawing one row per
    unique experimental combination at each epoch.  This implementation keeps
    ``ConcatDataset`` support while avoiding a fixed ``Subset`` so that repeated
    measurements can be resampled across epochs.
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

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        sampler = BalancedCombinationSampler(
            train_dataset,
            num_samples_per_combination=self.num_samples_per_combination,
            seed=self.random_seed,
            shuffle=True,
        )
        batch_sampler = BatchSampler(
            sampler,
            self.args.train_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )
        return DataLoader(
            train_dataset,
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
    # These small scalar / near-identity parameters have explicit losses or
    # learned scaling semantics.  Applying AdamW weight decay would pull them
    # toward zero rather than toward their intended neutral values.
    explicit_no_weight_decay = (
        "embed_scale",
        "reference_transform",
        "reference_log_scale",
        "reference_shift",
    )
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
                p
                for n, p in other_params
                if not any(nd in n for nd in no_decay)
                and not any(key in n for key in explicit_no_weight_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in other_params
                if any(nd in n for nd in no_decay)
                or any(key in n for key in explicit_no_weight_decay)
            ],
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
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> np.ndarray:
    """Compute frozen ESM embedding distances for paired examples."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mixed_dtype = _mixed_precision_dtype(
        device, use_bf16=use_bf16, use_fp16=use_fp16
    )

    config = EsmConfig.from_pretrained(esm_model_name)
    model = ESMEmbeddingDistanceModel(config, esm_model_name)
    if mixed_dtype is not None:
        model = model.to(device=device, dtype=mixed_dtype)
    else:
        model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    distances = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=mixed_dtype)
        if mixed_dtype is not None and device.type == "cuda"
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
    use_bf16: bool = True,
    use_fp16: bool = False,
) -> float:
    """Estimate the semantic-loss scale factor used in PLANT training."""
    distances = compute_embedding_distances(
        dataset,
        esm_model_name=esm_model_name,
        batch_size=batch_size,
        device=device,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
    )
    return float(np.quantile(distances, quantile))

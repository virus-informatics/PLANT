"""PLANT model class.

The class is usable both for inference (sequence -> antigenic-map coordinate) and
for training with paired antigenic-distance data plus optional virus-only
semantic regularization data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import AutoModel, EsmConfig, PreTrainedModel
from transformers.utils import ModelOutput


OHE_virus = None
OHE_ref = None
OHE_vp = None
OHE_rp = None

MISSING_LABEL_VALUE = -10.0


def set_encoders(ohe_virus=None, ohe_ref=None, ohe_vp=None, ohe_rp=None) -> None:
    """Register fitted OneHotEncoder objects used by systematic-error terms."""
    global OHE_virus, OHE_ref, OHE_vp, OHE_rp
    OHE_virus = ohe_virus
    OHE_ref = ohe_ref
    OHE_vp = ohe_vp
    OHE_rp = ohe_rp


def _safe_category_count(encoder) -> int:
    if encoder is None:
        return 0
    return len(encoder.categories_[0])


class semanticESM(PreTrainedModel):
    """PLANT/pLM-DMS model.

    During training, examples with ``labels == -10`` are treated as virus-only
    examples and contribute only to the semantic regularization loss.  Examples
    with real labels contribute to the supervised antigenic-distance loss plus
    the semantic and local/global regularizers.
    """

    config_class = EsmConfig

    def __init__(
        self,
        config: EsmConfig,
        esm_model_name: str,
        effects_len: Optional[int] = None,
        virus_effects_len: Optional[int] = None,
        embed_scale_factor: float = 1.0,
        latent_dim: int = 3,
        intermediate_dim: int = 256,
        intermediate_dim_encoder: int = 64,
        dropout: float = 0.05,
        dropout_encoder: float = 0.1,
        MAIN_W: float = 1.0,
        CSE_W: float = 0.0,
        CSE_ALPHA: float = 0.0,
        SEMANTIC_W: float = 0.2,
        CSE_W_VIRUS_ONLY: float = 0.0,
        SEMANTIC_W_VIRUS_ONLY: float = 0.2,
        CART_W: float = 0.05,
        LG_W: float = 0.01,
        missing_label_value: float = MISSING_LABEL_VALUE,
    ) -> None:
        super().__init__(config)

        if virus_effects_len is None:
            virus_effects_len = _safe_category_count(OHE_virus)
        if effects_len is None:
            effects_len = (
                _safe_category_count(OHE_ref)
                + _safe_category_count(OHE_vp)
                + _safe_category_count(OHE_rp)
            )

        self.esm_model = AutoModel.from_pretrained(
            esm_model_name, add_pooling_layer=False
        )
        self.esm_model_original = self._initialize_frozen_esm_model(esm_model_name)
        self.embedding_dim = self.esm_model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim),
        )

        self.virus_effects = (
            nn.Linear(virus_effects_len, 1, bias=False)
            if virus_effects_len > 0
            else None
        )
        self.systematic_error_effects = (
            nn.Sequential(
                nn.Linear(effects_len, intermediate_dim_encoder),
                nn.ReLU(),
                nn.Dropout(dropout_encoder),
                nn.Linear(intermediate_dim_encoder, 1, bias=False),
            )
            if effects_len > 0
            else None
        )

        self.embed_scale = nn.Parameter(torch.tensor(1.0))
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_mean = nn.MSELoss(reduction="none")

        self.MAIN_W = MAIN_W
        self.CSE_W = CSE_W
        self.SEMANTIC_W = SEMANTIC_W
        self.CSE_ALPHA = CSE_ALPHA
        self.CART_W = CART_W
        self.CSE_W_VIRUS_ONLY = CSE_W_VIRUS_ONLY
        self.SEMANTIC_W_VIRUS_ONLY = SEMANTIC_W_VIRUS_ONLY
        self.LG_W = LG_W
        self.embed_scale_factor = float(embed_scale_factor)
        self.missing_label_value = float(missing_label_value)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _initialize_frozen_esm_model(self, esm_model_name: str):
        esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        for param in esm_model.parameters():
            param.requires_grad = False
        esm_model.eval()
        return esm_model

    def train(self, mode: bool = True):  # noqa: D401
        """Set training mode while keeping ``esm_model_original`` in eval mode."""
        super().train(mode)
        self.esm_model_original.eval()
        return self

    def encode_sequence(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_out = model(
            input_ids.to(self.device), attention_mask=attention_mask.to(self.device)
        )
        return encoder_out.last_hidden_state[:, 0, :]

    def encode_one_hot(self, encoder, input_tensor: Optional[torch.Tensor]):
        if encoder is None or input_tensor is None:
            return None
        model_dtype = next(self.parameters()).dtype
        encoded = encoder.transform(
            input_tensor.detach().cpu().numpy().reshape(-1, 1)
        ).toarray()
        return torch.tensor(encoded, dtype=model_dtype, device=self.device)

    def extract_pooled_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract frozen ESM mean+max pooled embeddings for semantic loss."""
        model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            encoder_out = model(input_ids, attention_mask=attention_mask).last_hidden_state
            masked_sum = (encoder_out * attention_mask.unsqueeze(-1)).sum(dim=1)
            mask_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pooled = masked_sum / mask_count

            # Avoid padded tokens dominating max pooling.
            mask = attention_mask.unsqueeze(-1).bool()
            masked_encoder_out = encoder_out.masked_fill(~mask, torch.finfo(encoder_out.dtype).min)
            max_pooled = torch.max(masked_encoder_out, dim=1)[0]
            return torch.cat([mean_pooled, max_pooled], dim=-1)

    def compute_semantic_loss(
        self,
        latents: torch.Tensor,
        latents_original: torch.Tensor,
        embed_scale: torch.Tensor,
        embed_scale_factor: float,
    ) -> torch.Tensor:
        if latents.size(0) < 2:
            return torch.tensor(0.0, device=latents.device)
        pairwise_distances = torch.cdist(latents, latents, p=2)
        upper_triangle_indices = torch.triu_indices(
            pairwise_distances.size(0),
            pairwise_distances.size(1),
            offset=1,
            device=latents.device,
        )
        pairwise_distances_original = torch.cdist(latents_original, latents_original, p=2)
        pairwise_distances_original = pairwise_distances_original / embed_scale_factor
        return self.mse_loss_wo_mean(
            pairwise_distances_original[upper_triangle_indices[0], upper_triangle_indices[1]],
            pairwise_distances[upper_triangle_indices[0], upper_triangle_indices[1]]
            * embed_scale,
        ).mean()

    def contrastive_loss_semantic(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        margin: float = 1.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        if embeddings1.size(0) < 2:
            return torch.tensor(0.0, device=embeddings1.device)
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)
        positive_loss = torch.mean(torch.diag(distance_matrix))
        margin_loss = torch.clamp(margin - distance_matrix, min=0)
        weight = torch.exp(-alpha * distance_matrix)
        negative_loss = torch.mean(weight * margin_loss)
        return positive_loss + negative_loss

    def local_global_loss(
        self,
        latents: torch.Tensor,
        k_local: int = 3,
        margin_global: float = 0.125,
    ) -> torch.Tensor:
        if latents.size(0) < 2:
            return torch.tensor(0.0, device=latents.device)
        dists = torch.cdist(latents, latents, p=2)
        n = latents.size(0)
        eye_mask = torch.eye(n, device=dists.device).bool()
        dists_no_self = dists.masked_fill(eye_mask, float("inf"))

        k_safe = min(k_local, n - 1)
        if k_safe > 0:
            knn_dists, _ = torch.topk(dists_no_self, k=k_safe, largest=False, dim=1)
            local_loss = torch.mean(knn_dists)
        else:
            local_loss = torch.tensor(0.0, device=latents.device)

        margin_mask = dists_no_self < margin_global
        repel_loss = torch.clamp(margin_global - dists_no_self, min=0.0)
        global_loss = torch.sum(repel_loss * margin_mask) / (margin_mask.sum() + 1e-8)
        return local_loss + global_loss

    def custom_loss(
        self,
        predictions: torch.Tensor,
        predictions_cart: torch.Tensor,
        targets: torch.Tensor,
        censors: torch.Tensor,
        virus_regressor_out: torch.Tensor,
        virus_regressor_out2: torch.Tensor,
        reference_regressor_out: torch.Tensor,
        reference_regressor_out2: torch.Tensor,
        virus_embedding_original: torch.Tensor,
        reference_embedding_original: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        uncensored_loss = self.mse_loss_wo_mean(predictions, targets) * (1 - censors)
        censored_loss = self.mse_loss_wo_mean(
            predictions, torch.maximum(predictions, targets)
        ) * censors
        uncensored_loss_cart = self.mse_loss_wo_mean(predictions_cart, targets) * (
            1 - censors
        )
        censored_loss_cart = self.mse_loss_wo_mean(
            predictions_cart, torch.maximum(predictions_cart, targets)
        ) * censors

        uncensored_loss = uncensored_loss * weight
        censored_loss = censored_loss * weight
        uncensored_loss_cart = uncensored_loss_cart * weight
        censored_loss_cart = censored_loss_cart * weight

        combined_latents = torch.cat([virus_regressor_out, reference_regressor_out], dim=0)
        combined_latents2 = torch.cat([virus_regressor_out2, reference_regressor_out2], dim=0)
        combined_latents_original = torch.cat(
            [virus_embedding_original, reference_embedding_original], dim=0
        )

        contrastive_loss_value = self.contrastive_loss_semantic(
            combined_latents, combined_latents2, alpha=self.CSE_ALPHA
        )
        local_global_loss_value = self.local_global_loss(
            combined_latents, k_local=3, margin_global=0.125
        )
        semantic_loss = self.compute_semantic_loss(
            combined_latents,
            combined_latents_original,
            self.embed_scale,
            self.embed_scale_factor,
        )

        return (
            torch.mean(uncensored_loss) * self.MAIN_W
            + torch.mean(censored_loss) * self.MAIN_W
            + torch.mean(uncensored_loss_cart) * self.CART_W
            + torch.mean(censored_loss_cart) * self.CART_W
            + contrastive_loss_value * self.CSE_W
            + semantic_loss * self.SEMANTIC_W
            + local_global_loss_value * self.LG_W
        )

    def custom_loss_only_semantic(
        self,
        virus_regressor_out: torch.Tensor,
        virus_regressor_out2: torch.Tensor,
        virus_embedding_original: torch.Tensor,
    ) -> torch.Tensor:
        contrastive_loss_value = self.contrastive_loss_semantic(
            virus_regressor_out, virus_regressor_out2, alpha=self.CSE_ALPHA
        )
        local_global_loss_value = self.local_global_loss(
            virus_regressor_out, k_local=3, margin_global=0.125
        )
        semantic_loss = self.compute_semantic_loss(
            virus_regressor_out,
            virus_embedding_original,
            self.embed_scale,
            self.embed_scale_factor,
        )
        return (
            semantic_loss * self.SEMANTIC_W_VIRUS_ONLY
            + contrastive_loss_value * self.CSE_W_VIRUS_ONLY
            + local_global_loss_value * self.LG_W
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        **kwargs,
    ):
        """Load a saved PLANT checkpoint.

        This keeps compatibility with the existing local safetensors checkpoint
        layout while also accepting both single-file and sharded safetensors.
        """
        if config is None:
            config = EsmConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, *model_args, **kwargs)
        checkpoint_dir = Path(pretrained_model_name_or_path)

        safetensor_files = []
        single_file = checkpoint_dir / "model.safetensors"
        if single_file.exists():
            safetensor_files.append(single_file)
        safetensor_files.extend(sorted(checkpoint_dir.glob("model-*-of-*.safetensors")))

        if not safetensor_files:
            raise FileNotFoundError(
                f"No model.safetensors or model-*-of-*.safetensors files found in "
                f"{checkpoint_dir}."
            )

        state_dict = {}
        for f in safetensor_files:
            part = safe_load(str(f), device="cpu")
            state_dict.update(part)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print("[PLANT] Missing keys:", missing_keys)
        if unexpected_keys:
            print("[PLANT] Unexpected keys:", unexpected_keys)
        return model

    def forward(
        self,
        input_ids_virus: torch.Tensor,
        attention_mask_virus: torch.Tensor,
        input_ids_reference: Optional[torch.Tensor] = None,
        attention_mask_reference: Optional[torch.Tensor] = None,
        censors: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        virus: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        dates: Optional[torch.Tensor] = None,
        virus_passage: Optional[torch.Tensor] = None,
        reference_passage: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        del dates, kwargs  # currently unused, kept for API compatibility
        device = self.device
        input_ids_virus = input_ids_virus.to(device)
        attention_mask_virus = attention_mask_virus.to(device)

        virus_regressor_out = self.regressor(
            self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus)
        )
        virus_regressor_out2 = self.regressor(
            self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus)
        )

        # Fast path for pure inference: sequence -> PLANT coordinates.
        if labels is None and input_ids_reference is None:
            return ModelOutput(
                loss=torch.tensor(0.0, device=device),
                logits=None,
                hidden_state_virus=virus_regressor_out,
            )

        batch_size = input_ids_virus.size(0)
        if labels is None:
            has_labels_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            labels = labels.to(device).view(-1)
            has_labels_mask = labels.ne(self.missing_label_value)

        if input_ids_reference is None:
            input_ids_reference = torch.zeros_like(input_ids_virus)
        if attention_mask_reference is None:
            attention_mask_reference = torch.zeros_like(attention_mask_virus)
        input_ids_reference = input_ids_reference.to(device)
        attention_mask_reference = attention_mask_reference.to(device)

        virus_embedding_original = self.extract_pooled_embeddings(
            self.esm_model_original, input_ids_virus, attention_mask_virus
        )

        # Semantic-only loss for virus-only examples.
        no_label_mask = ~has_labels_mask
        if no_label_mask.any():
            loss_no_labels = self.custom_loss_only_semantic(
                virus_regressor_out[no_label_mask],
                virus_regressor_out2[no_label_mask],
                virus_embedding_original[no_label_mask],
            )
        else:
            loss_no_labels = torch.tensor(0.0, device=device)

        # Supervised antigenic-distance loss for paired examples.
        if has_labels_mask.any():
            input_ids_reference_labeled = input_ids_reference[has_labels_mask]
            attention_mask_reference_labeled = attention_mask_reference[has_labels_mask]

            reference_regressor_out = self.regressor(
                self.encode_sequence(
                    self.esm_model,
                    input_ids_reference_labeled,
                    attention_mask_reference_labeled,
                )
            )
            reference_regressor_out2 = self.regressor(
                self.encode_sequence(
                    self.esm_model,
                    input_ids_reference_labeled,
                    attention_mask_reference_labeled,
                )
            )
            distance = torch.norm(
                virus_regressor_out[has_labels_mask] - reference_regressor_out,
                p=2,
                dim=1,
                keepdim=True,
            )

            systematic_error = torch.zeros_like(distance)
            systematic_error1 = torch.zeros_like(distance)
            systematic_error2 = torch.zeros_like(distance)

            if self.virus_effects is not None:
                virus_encoding = self.encode_one_hot(
                    OHE_virus,
                    virus.to(device)[has_labels_mask] if virus is not None else None,
                )
                if virus_encoding is not None:
                    systematic_error1 = self.virus_effects(virus_encoding)
                    systematic_error = systematic_error + systematic_error1

            if self.systematic_error_effects is not None:
                parts = []
                if reference is not None:
                    x = self.encode_one_hot(OHE_ref, reference.to(device)[has_labels_mask])
                    if x is not None:
                        parts.append(x)
                if virus_passage is not None:
                    x = self.encode_one_hot(OHE_vp, virus_passage.to(device)[has_labels_mask])
                    if x is not None:
                        parts.append(x)
                if reference_passage is not None:
                    x = self.encode_one_hot(
                        OHE_rp, reference_passage.to(device)[has_labels_mask]
                    )
                    if x is not None:
                        parts.append(x)
                if parts:
                    combined_encoding = torch.cat(parts, dim=-1)
                    systematic_error2 = self.systematic_error_effects(combined_encoding)
                    systematic_error = systematic_error + systematic_error2

            observed_distance = distance + systematic_error
            logits_labeled = torch.cat((observed_distance, distance), dim=1)

            reference_embedding_original = self.extract_pooled_embeddings(
                self.esm_model_original,
                input_ids_reference_labeled,
                attention_mask_reference_labeled,
            )

            if censors is None:
                censors = torch.zeros(batch_size, dtype=torch.float, device=device)
            if weight is None:
                weight = torch.ones(batch_size, dtype=torch.float, device=device)

            combined_loss = self.custom_loss(
                observed_distance,
                distance,
                labels[has_labels_mask].view(-1, 1),
                censors.to(device)[has_labels_mask].view(-1, 1),
                virus_regressor_out[has_labels_mask],
                virus_regressor_out2[has_labels_mask],
                reference_regressor_out,
                reference_regressor_out2,
                virus_embedding_original[has_labels_mask],
                reference_embedding_original,
                weight.to(device)[has_labels_mask].view(-1, 1),
            )
            combined_loss = combined_loss + (
                torch.mean(systematic_error1**2) + torch.mean(systematic_error2**2)
            ) * 1.0e-4

            # Trainer.predict expects batch-aligned outputs.  Virus-only rows get NaN.
            logits = torch.full(
                (batch_size, 2), float("nan"), device=device, dtype=logits_labeled.dtype
            )
            logits[has_labels_mask] = logits_labeled
        else:
            combined_loss = torch.tensor(0.0, device=device)
            logits = None

        total_loss = combined_loss + loss_no_labels
        return ModelOutput(
            loss=total_loss,
            logits=logits,
            hidden_state_virus=virus_regressor_out,
        )

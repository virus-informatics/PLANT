"""PLANT model class.

The class is usable both for inference (sequence -> antigenic-map coordinate) and
for training with paired antigenic-distance data plus optional virus-only
semantic regularization data.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import AutoModel, EsmConfig, PreTrainedModel
from transformers.utils import ModelOutput

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - PEFT is optional unless use_lora=True.
    LoraConfig = None
    get_peft_model = None


MISSING_LABEL_VALUE = -10.0
PLANT_INIT_CONFIG_NAME = "plant_model_config.json"


def _safe_category_count(encoder) -> int:
    if encoder is None:
        return 0
    return len(encoder.categories_[0])


class semanticESM(PreTrainedModel):
    """PLANT/pLM-DMS model.

    During training, examples with ``labels == -10`` are treated as virus-only
    examples. Supervised antigenic-distance loss is computed only for labeled
    pairs, while semantic, contrastive (CSE), and local/global regularization are
    computed once over the unified set of paired viruses, paired references, and
    virus-only sequences present in the mini-batch.
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
        SEMANTIC_W: float = 0.1,
        # Deprecated compatibility arguments. Unified regularization uses CSE_W
        # and SEMANTIC_W for every sequence in the mini-batch.
        CSE_W_VIRUS_ONLY: Optional[float] = None,
        SEMANTIC_W_VIRUS_ONLY: Optional[float] = None,
        CART_W: float = 0.1,
        LG_W: float = 0.0,
        reference_transform_mode: str = "none",
        REF_TRANSFORM_W: float = 0.05,
        REF_SHIFT_W: float = 0.05,
        missing_label_value: float = MISSING_LABEL_VALUE,
        use_systematic_error: bool = True,
        freeze_esm: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[Sequence[str]] = None,
        lora_bias: str = "none",
    ) -> None:
        super().__init__(config)

        # Encoder objects are attached to each model instance with set_encoders().
        # The layer dimensions must therefore be supplied explicitly when systematic
        # error terms are used. Saved checkpoints contain these values in
        # plant_model_config.json.
        if virus_effects_len is None:
            virus_effects_len = 0
        if effects_len is None:
            effects_len = 0

        self.esm_model_name = str(esm_model_name)
        self.effects_len = int(effects_len) if effects_len is not None else None
        self.virus_effects_len = (
            int(virus_effects_len) if virus_effects_len is not None else None
        )
        self.latent_dim = int(latent_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.intermediate_dim_encoder = int(intermediate_dim_encoder)
        self.dropout = float(dropout)
        self.dropout_encoder = float(dropout_encoder)
        self.freeze_esm = bool(freeze_esm)
        self.use_lora = bool(use_lora)
        if self.freeze_esm and self.use_lora:
            raise ValueError("freeze_esm=True and use_lora=True cannot be set simultaneously.")
        if lora_target_modules is None:
            self.lora_target_modules = ["query", "key", "value"]
        else:
            self.lora_target_modules = list(lora_target_modules)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_bias = str(lora_bias)

        base_esm_model = AutoModel.from_pretrained(
            esm_model_name, add_pooling_layer=False
        )
        self.embedding_dim = base_esm_model.config.hidden_size

        if self.freeze_esm:
            for param in base_esm_model.parameters():
                param.requires_grad = False
            # A newly constructed nn.Module starts in training mode. Freeze mode
            # should be deterministic even before the outer PLANT model receives
            # its first explicit train()/eval() call.
            base_esm_model.eval()
            self.esm_model = base_esm_model
            self.esm_model_original = None
        elif self.use_lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError(
                    "use_lora=True requires the `peft` package. Install it with `pip install peft`."
                )
            for param in base_esm_model.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=lora_dropout,
                bias=lora_bias,
            )
            self.esm_model = get_peft_model(base_esm_model, lora_config)
            # In LoRA mode, the original frozen ESM is obtained by temporarily
            # disabling adapters on this single shared backbone.
            self.esm_model_original = None
        else:
            self.esm_model = base_esm_model
            self.esm_model_original = self._initialize_frozen_esm_model(esm_model_name)

        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim),
        )

        self.reference_transform_mode = reference_transform_mode.lower()
        valid_reference_transform_modes = {"none", "full", "diagonal"}
        if self.reference_transform_mode not in valid_reference_transform_modes:
            raise ValueError(
                "reference_transform_mode must be one of "
                f"{sorted(valid_reference_transform_modes)}, got "
                f"{reference_transform_mode!r}."
            )

        # Optional near-identity transform for the reference/serum-side coordinate.
        # This allows the reference coordinate system to differ slightly from the
        # target-virus coordinate system while starting from the original PLANT
        # behavior: z_r_final = z_r.
        self.reference_transform = None
        self.reference_log_scale = None
        self.reference_shift = None
        if self.reference_transform_mode == "full":
            self.reference_transform = nn.Linear(latent_dim, latent_dim, bias=True)
            with torch.no_grad():
                self.reference_transform.weight.copy_(torch.eye(latent_dim))
                self.reference_transform.bias.zero_()
        elif self.reference_transform_mode == "diagonal":
            self.reference_log_scale = nn.Parameter(torch.zeros(latent_dim))
            self.reference_shift = nn.Parameter(torch.zeros(latent_dim))

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
        # Accept legacy checkpoint fields without retaining two independent sets
        # of regularization weights. They are intentionally ignored.
        del CSE_W_VIRUS_ONLY, SEMANTIC_W_VIRUS_ONLY
        self.LG_W = LG_W
        self.REF_TRANSFORM_W = REF_TRANSFORM_W
        self.REF_SHIFT_W = REF_SHIFT_W
        self.embed_scale_factor = float(embed_scale_factor)
        self.missing_label_value = float(missing_label_value)
        # Model-level default used by Trainer, which does not pass a constant
        # apply_systematic_error argument with every mini-batch. The forward
        # argument can still explicitly override this default during inference.
        self.use_systematic_error = bool(use_systematic_error)

        # OneHotEncoder objects are runtime artifacts and are intentionally kept on
        # the model instance rather than in module-global state. This allows multiple
        # PLANT models with different category spaces to coexist safely in one process.
        self.ohe_virus = None
        self.ohe_ref = None
        self.ohe_vp = None
        self.ohe_rp = None

    def set_encoders(self, ohe_virus=None, ohe_ref=None, ohe_vp=None, ohe_rp=None) -> None:
        """Attach fitted systematic-error encoders to this model instance.

        The encoder dimensions are checked against the trained linear layers when
        an encoder is provided. Missing encoders are allowed for coordinate-only
        inference, where ``apply_systematic_error=False`` is used.
        """
        if ohe_virus is not None and self.virus_effects is not None:
            actual = _safe_category_count(ohe_virus)
            expected = self.virus_effects.in_features
            if actual != expected:
                raise ValueError(
                    "Virus encoder category count does not match the model: "
                    f"encoder={actual}, model={expected}."
                )

        provided_effect_encoders = [ohe_ref, ohe_vp, ohe_rp]
        n_provided_effect_encoders = sum(
            encoder is not None for encoder in provided_effect_encoders
        )
        if n_provided_effect_encoders not in {0, len(provided_effect_encoders)}:
            raise ValueError(
                "Reference, virus-passage, and reference-passage encoders must "
                "either all be provided or all be omitted."
            )

        if (
            self.systematic_error_effects is not None
            and n_provided_effect_encoders == len(provided_effect_encoders)
        ):
            actual = sum(_safe_category_count(encoder) for encoder in provided_effect_encoders)
            expected = self.systematic_error_effects[0].in_features
            if actual != expected:
                raise ValueError(
                    "Reference/passage encoder category count does not match the model: "
                    f"encoders={actual}, model={expected}."
                )

        self.ohe_virus = ohe_virus
        self.ohe_ref = ohe_ref
        self.ohe_vp = ohe_vp
        self.ohe_rp = ohe_rp

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
        """Set training mode while keeping frozen ESM paths in evaluation mode."""
        super().train(mode)
        if self.freeze_esm:
            # requires_grad=False does not disable dropout. Keep the frozen backbone
            # deterministic throughout regressor-only training.
            self.esm_model.eval()
        if self.esm_model_original is not None:
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

    def _adapter_disabled_context(self):
        """Return a context in which PEFT adapters are disabled, if available."""
        if self.use_lora and hasattr(self.esm_model, "disable_adapter"):
            return self.esm_model.disable_adapter()
        return contextlib.nullcontext()

    def extract_pooled_embeddings(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract frozen ESM mean+max pooled embeddings for semantic loss."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                encoder_out = model(input_ids, attention_mask=attention_mask).last_hidden_state
                masked_sum = (encoder_out * attention_mask.unsqueeze(-1)).sum(dim=1)
                mask_count = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean_pooled = masked_sum / mask_count

                # Avoid padded tokens dominating max pooling.
                mask = attention_mask.unsqueeze(-1).bool()
                masked_encoder_out = encoder_out.masked_fill(
                    ~mask, torch.finfo(encoder_out.dtype).min
                )
                max_pooled = torch.max(masked_encoder_out, dim=1)[0]
                return torch.cat([mean_pooled, max_pooled], dim=-1)
        finally:
            if was_training:
                model.train()

    def extract_original_pooled_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract original frozen ESM embeddings.

        In freeze ESM mode (no LoRA, no full-finetuning) this uses the frozen ESM.
        In full-finetuning mode this uses the separate frozen ESM copy.
        In LoRA mode this reuses the single ESM backbone with adapters temporarily
        disabled, avoiding a second copy of ESM-2 in memory.
        """
        if self.freeze_esm:
            return self.extract_pooled_embeddings(
                self.esm_model, input_ids, attention_mask
            )
        if self.use_lora:
            with self._adapter_disabled_context():
                return self.extract_pooled_embeddings(
                    self.esm_model, input_ids, attention_mask
                )
        if self.esm_model_original is None:
            raise RuntimeError("esm_model_original is not initialized.")
        return self.extract_pooled_embeddings(
            self.esm_model_original, input_ids, attention_mask
        )

    def compute_semantic_loss(
        self,
        latents: torch.Tensor,
        latents_original: torch.Tensor,
        embed_scale: torch.Tensor,
        embed_scale_factor: float,
    ) -> torch.Tensor:
        if latents.size(0) < 2:
            return latents.sum() * 0.0
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
        same_sequence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Contrast two stochastic views without treating duplicate sequences as negatives."""
        if embeddings1.size(0) < 2:
            return embeddings1.sum() * 0.0
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)

        if same_sequence_mask is None:
            if distance_matrix.size(0) != distance_matrix.size(1):
                raise ValueError(
                    "same_sequence_mask is required for non-square CSE distance matrices."
                )
            same_sequence_mask = torch.eye(
                distance_matrix.size(0),
                device=distance_matrix.device,
                dtype=torch.bool,
            )
        else:
            same_sequence_mask = same_sequence_mask.to(
                device=distance_matrix.device,
                dtype=torch.bool,
            )
            if same_sequence_mask.shape != distance_matrix.shape:
                raise ValueError(
                    "same_sequence_mask shape must match the CSE distance matrix: "
                    f"{tuple(same_sequence_mask.shape)} != {tuple(distance_matrix.shape)}."
                )

        positive_distances = distance_matrix[same_sequence_mask]
        if positive_distances.numel() == 0:
            positive_loss = embeddings1.sum() * 0.0
        else:
            positive_loss = torch.mean(positive_distances)

        negative_distances = distance_matrix[~same_sequence_mask]
        if negative_distances.numel() == 0:
            negative_loss = embeddings1.sum() * 0.0
        else:
            margin_loss = torch.clamp(margin - negative_distances, min=0)
            weight = torch.exp(-alpha * negative_distances)
            negative_loss = torch.mean(weight * margin_loss)
        return positive_loss + negative_loss

    def local_global_loss(
        self,
        latents: torch.Tensor,
        k_local: int = 3,
        margin_global: float = 0.125,
        same_sequence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents.size(0) < 2:
            return latents.sum() * 0.0
        dists = torch.cdist(latents, latents, p=2)
        n = latents.size(0)
        eye_mask = torch.eye(n, device=dists.device).bool()
        dists_no_self = dists.masked_fill(eye_mask, float("inf"))

        k_safe = min(k_local, n - 1)
        if k_safe > 0:
            knn_dists, _ = torch.topk(dists_no_self, k=k_safe, largest=False, dim=1)
            local_loss = torch.mean(knn_dists)
        else:
            local_loss = latents.sum() * 0.0

        if same_sequence_mask is None:
            same_sequence_mask = eye_mask
        else:
            same_sequence_mask = same_sequence_mask.to(
                device=dists.device,
                dtype=torch.bool,
            )
            if same_sequence_mask.shape != dists.shape:
                raise ValueError(
                    "same_sequence_mask shape must match the LG distance matrix: "
                    f"{tuple(same_sequence_mask.shape)} != {tuple(dists.shape)}."
                )
            same_sequence_mask = same_sequence_mask | eye_mask

        # Identical sequences may occur repeatedly as references or in both the
        # paired and virus-only pools. They must not repel each other globally.
        margin_mask = (dists_no_self < margin_global) & ~same_sequence_mask
        repel_loss = torch.clamp(margin_global - dists_no_self, min=0.0)
        global_loss = torch.sum(repel_loss * margin_mask) / (margin_mask.sum() + 1e-8)
        return local_loss + global_loss

    def apply_reference_transform(self, reference_latents: torch.Tensor) -> torch.Tensor:
        """Map shared reference coordinates to the effective reference coordinate space."""
        if self.reference_transform_mode == "none":
            return reference_latents
        if self.reference_transform_mode == "full":
            if self.reference_transform is None:
                raise RuntimeError("reference_transform is not initialized.")
            return self.reference_transform(reference_latents)
        if self.reference_transform_mode == "diagonal":
            if self.reference_log_scale is None or self.reference_shift is None:
                raise RuntimeError("diagonal reference transform is not initialized.")
            scale = torch.exp(self.reference_log_scale).to(
                device=reference_latents.device,
                dtype=reference_latents.dtype,
            )
            shift = self.reference_shift.to(
                device=reference_latents.device,
                dtype=reference_latents.dtype,
            )
            return reference_latents * scale + shift
        raise RuntimeError(f"Unknown reference_transform_mode: {self.reference_transform_mode}")

    def reference_transform_regularization(
        self,
        reference_latents_shared: torch.Tensor,
        reference_latents_final: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the reference coordinate transform close to the shared coordinate system."""
        if self.reference_transform_mode == "none":
            return reference_latents_shared.new_tensor(0.0)

        loss = reference_latents_shared.new_tensor(0.0)

        if self.REF_SHIFT_W != 0.0:
            # Data-scale regularization: penalize how much reference points move.
            shift_loss = torch.mean((reference_latents_final - reference_latents_shared) ** 2)
            loss = loss + shift_loss * self.REF_SHIFT_W

        if self.REF_TRANSFORM_W != 0.0:
            if self.reference_transform_mode == "full":
                if self.reference_transform is None:
                    raise RuntimeError("reference_transform is not initialized.")
                eye = torch.eye(
                    self.reference_transform.weight.size(0),
                    device=self.reference_transform.weight.device,
                    dtype=self.reference_transform.weight.dtype,
                )
                transform_loss = torch.mean((self.reference_transform.weight - eye) ** 2)
            elif self.reference_transform_mode == "diagonal":
                if self.reference_log_scale is None:
                    raise RuntimeError("diagonal reference transform is not initialized.")
                scale = torch.exp(self.reference_log_scale)
                transform_loss = torch.mean((scale - 1.0) ** 2)
            else:
                transform_loss = reference_latents_shared.new_tensor(0.0)
            loss = loss + transform_loss * self.REF_TRANSFORM_W

        return loss

    def custom_loss(
        self,
        predictions: torch.Tensor,
        predictions_cart: torch.Tensor,
        targets: torch.Tensor,
        censors: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised censored regression loss for labeled antigenic pairs."""
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

        return (
            torch.mean(uncensored_loss) * self.MAIN_W
            + torch.mean(censored_loss) * self.MAIN_W
            + torch.mean(uncensored_loss_cart) * self.CART_W
            + torch.mean(censored_loss_cart) * self.CART_W
        )

    def compute_unified_regularization_loss(
        self,
        latents_view1: torch.Tensor,
        *,
        latents_original: Optional[torch.Tensor] = None,
        latents_view2: Optional[torch.Tensor] = None,
        same_sequence_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Regularize one unified sequence set exactly once per mini-batch.

        ``latents_view1`` contains every virus input in the batch (paired and
        virus-only) followed, when present, by the labeled reference sequences.
        Reference coordinates are supplied in the shared sequence space before
        any serum-side reference transform is applied.
        """
        zero = latents_view1.sum() * 0.0
        total = zero

        if self.SEMANTIC_W != 0.0:
            if latents_original is None:
                raise ValueError(
                    "latents_original is required when SEMANTIC_W is non-zero."
                )
            semantic_loss = self.compute_semantic_loss(
                latents_view1,
                latents_original,
                self.embed_scale,
                self.embed_scale_factor,
            )
            total = total + semantic_loss * self.SEMANTIC_W

        if self.CSE_W != 0.0:
            if latents_view2 is None:
                raise ValueError("latents_view2 is required when CSE_W is non-zero.")
            cse_loss = self.contrastive_loss_semantic(
                latents_view1,
                latents_view2,
                alpha=self.CSE_ALPHA,
                same_sequence_mask=same_sequence_mask,
            )
            total = total + cse_loss * self.CSE_W

        if self.LG_W != 0.0:
            local_global_loss = self.local_global_loss(
                latents_view1,
                k_local=3,
                margin_global=0.125,
                same_sequence_mask=same_sequence_mask,
            )
            total = total + local_global_loss * self.LG_W

        return total

    def get_plant_init_config(self) -> dict:
        """Return constructor arguments needed to reload this PLANT model."""
        return {
            "esm_model_name": self.esm_model_name,
            "effects_len": self.effects_len,
            "virus_effects_len": self.virus_effects_len,
            "embed_scale_factor": self.embed_scale_factor,
            "latent_dim": self.latent_dim,
            "intermediate_dim": self.intermediate_dim,
            "intermediate_dim_encoder": self.intermediate_dim_encoder,
            "dropout": self.dropout,
            "dropout_encoder": self.dropout_encoder,
            "MAIN_W": self.MAIN_W,
            "CSE_W": self.CSE_W,
            "CSE_ALPHA": self.CSE_ALPHA,
            "SEMANTIC_W": self.SEMANTIC_W,
            "CART_W": self.CART_W,
            "LG_W": self.LG_W,
            "reference_transform_mode": self.reference_transform_mode,
            "REF_TRANSFORM_W": self.REF_TRANSFORM_W,
            "REF_SHIFT_W": self.REF_SHIFT_W,
            "missing_label_value": self.missing_label_value,
            "use_systematic_error": self.use_systematic_error,
            "freeze_esm": self.freeze_esm,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_bias": self.lora_bias,
        }

    def save_pretrained(self, save_directory, *args, **kwargs):  # noqa: D401
        """Save weights plus PLANT-specific constructor settings."""
        super().save_pretrained(save_directory, *args, **kwargs)
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        init_config_path = save_path / PLANT_INIT_CONFIG_NAME
        init_config_path.write_text(
            json.dumps(self.get_plant_init_config(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def _load_plant_init_config(cls, checkpoint_dir: Path) -> dict:
        init_config_path = checkpoint_dir / PLANT_INIT_CONFIG_NAME
        if not init_config_path.exists():
            return {}
        return json.loads(init_config_path.read_text(encoding="utf-8"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        strict: bool = True,
        **kwargs,
    ):
        """Load a saved PLANT checkpoint.

        If ``plant_model_config.json`` is present in the checkpoint directory,
        ``semanticESM.from_pretrained(model_dir)`` is sufficient to reconstruct
        the model architecture and PLANT-specific settings. Explicit keyword
        arguments override the saved settings. Positional ``model_args`` are
        kept for backward compatibility with older checkpoints.
        """
        checkpoint_dir = Path(pretrained_model_name_or_path)
        saved_init_kwargs = cls._load_plant_init_config(checkpoint_dir)

        if config is None:
            config = EsmConfig.from_pretrained(pretrained_model_name_or_path)

        init_kwargs = {**saved_init_kwargs, **kwargs}
        if model_args and "esm_model_name" in init_kwargs:
            # The first historical positional argument is esm_model_name.
            # Avoid passing it twice when loading with the legacy API.
            init_kwargs.pop("esm_model_name")

        if not model_args and "esm_model_name" not in init_kwargs:
            raise ValueError(
                f"Missing esm_model_name. Either save/load a checkpoint that contains "
                f"{PLANT_INIT_CONFIG_NAME}, or call from_pretrained(..., "
                "esm_model_name='<base ESM model>')."
            )

        model = cls(config, *model_args, **init_kwargs)

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
        if missing_keys or unexpected_keys:
            message = (
                "[PLANT] Checkpoint keys do not match the reconstructed model. "
                f"Missing keys: {missing_keys}; unexpected keys: {unexpected_keys}"
            )
            if strict:
                raise RuntimeError(message)
            print(message)

        # Match Hugging Face from_pretrained() semantics: returned models are in
        # evaluation mode unless the caller explicitly enables training.
        model.eval()
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
        apply_systematic_error: Optional[bool] = None,
        **kwargs,
    ) -> ModelOutput:
        del dates, kwargs  # currently unused, kept for API compatibility
        if apply_systematic_error is None:
            apply_systematic_error = self.use_systematic_error
        else:
            apply_systematic_error = bool(apply_systematic_error)

        device = self.device
        input_ids_virus = input_ids_virus.to(device)
        attention_mask_virus = attention_mask_virus.to(device)

        # This tensor contains both labeled paired viruses and virus-only rows.
        virus_regressor_out = self.regressor(
            self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus)
        )

        # Fast path for pure sequence -> coordinate inference.
        if labels is None and input_ids_reference is None:
            return ModelOutput(
                loss=torch.tensor(0.0, device=device),
                logits=None,
                hidden_state_virus=virus_regressor_out,
            )

        needs_cse = self.CSE_W != 0.0
        if needs_cse:
            virus_regressor_out2 = self.regressor(
                self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus)
            )
        else:
            virus_regressor_out2 = None

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

        # Unified regularization always starts with every virus row in the batch.
        regularization_latents_view1 = [virus_regressor_out]
        regularization_input_ids = [input_ids_virus]
        regularization_attention_masks = [attention_mask_virus]
        regularization_latents_view2 = (
            [virus_regressor_out2] if virus_regressor_out2 is not None else None
        )
        regularization_original = [] if self.SEMANTIC_W != 0.0 else None
        if regularization_original is not None:
            regularization_original.append(
                self.extract_original_pooled_embeddings(
                    input_ids_virus,
                    attention_mask_virus,
                )
            )

        # Supervised antigenic-distance loss is restricted to labeled pairs.
        if has_labels_mask.any():
            input_ids_reference_labeled = input_ids_reference[has_labels_mask]
            attention_mask_reference_labeled = attention_mask_reference[has_labels_mask]

            reference_regressor_out_shared = self.regressor(
                self.encode_sequence(
                    self.esm_model,
                    input_ids_reference_labeled,
                    attention_mask_reference_labeled,
                )
            )
            # Use the shared (pre-transform) reference coordinates for all sequence
            # regularizers, and only use the transformed coordinates for prediction.
            regularization_latents_view1.append(reference_regressor_out_shared)
            regularization_input_ids.append(input_ids_reference_labeled)
            regularization_attention_masks.append(attention_mask_reference_labeled)

            if needs_cse:
                reference_regressor_out2 = self.regressor(
                    self.encode_sequence(
                        self.esm_model,
                        input_ids_reference_labeled,
                        attention_mask_reference_labeled,
                    )
                )
                assert regularization_latents_view2 is not None
                regularization_latents_view2.append(reference_regressor_out2)

            if regularization_original is not None:
                regularization_original.append(
                    self.extract_original_pooled_embeddings(
                        input_ids_reference_labeled,
                        attention_mask_reference_labeled,
                    )
                )

            reference_regressor_out = self.apply_reference_transform(
                reference_regressor_out_shared
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

            if apply_systematic_error:
                if self.virus_effects is not None:
                    if self.ohe_virus is None or virus is None:
                        raise RuntimeError(
                            "Systematic-error prediction requires both the virus "
                            "encoder and virus category tensor. Attach encoders with "
                            "model.set_encoders(...) or set apply_systematic_error=False."
                        )
                    virus_encoding = self.encode_one_hot(
                        self.ohe_virus,
                        virus.to(device)[has_labels_mask],
                    )
                    if virus_encoding is not None:
                        systematic_error1 = self.virus_effects(virus_encoding)
                        systematic_error = systematic_error + systematic_error1

                if self.systematic_error_effects is not None:
                    effect_encoders = (self.ohe_ref, self.ohe_vp, self.ohe_rp)
                    effect_tensors = (reference, virus_passage, reference_passage)
                    if any(encoder is None for encoder in effect_encoders) or any(
                        tensor is None for tensor in effect_tensors
                    ):
                        raise RuntimeError(
                            "Systematic-error prediction requires reference, "
                            "virus-passage, and reference-passage encoders and category "
                            "tensors. Attach encoders with model.set_encoders(...) or "
                            "set apply_systematic_error=False."
                        )

                    parts = [
                        self.encode_one_hot(
                            encoder,
                            tensor.to(device)[has_labels_mask],
                        )
                        for encoder, tensor in zip(effect_encoders, effect_tensors)
                    ]
                    combined_encoding = torch.cat(parts, dim=-1)
                    systematic_error2 = self.systematic_error_effects(combined_encoding)
                    systematic_error = systematic_error + systematic_error2

            # Ablation mode is intentionally minimal: when systematic error is
            # disabled, the correction terms are not evaluated and remain zero, so
            # observed_distance is exactly the cartographic distance.
            observed_distance = distance + systematic_error
            logits_labeled = torch.cat((observed_distance, distance), dim=1)

            if censors is None:
                censors = torch.zeros(batch_size, dtype=torch.float, device=device)
            if weight is None:
                weight = torch.ones(batch_size, dtype=torch.float, device=device)

            supervised_loss = self.custom_loss(
                observed_distance,
                distance,
                labels[has_labels_mask].view(-1, 1),
                censors.to(device)[has_labels_mask].view(-1, 1),
                weight.to(device)[has_labels_mask].view(-1, 1),
            )
            supervised_loss = supervised_loss + (
                torch.mean(systematic_error1**2) + torch.mean(systematic_error2**2)
            ) * 1.0e-4
            supervised_loss = supervised_loss + self.reference_transform_regularization(
                reference_regressor_out_shared,
                reference_regressor_out,
            )

            # Trainer.predict expects batch-aligned outputs. Virus-only rows get NaN.
            logits = torch.full(
                (batch_size, 2),
                float("nan"),
                device=device,
                dtype=logits_labeled.dtype,
            )
            logits[has_labels_mask] = logits_labeled
        else:
            supervised_loss = virus_regressor_out.sum() * 0.0
            logits = None

        all_latents_view1 = torch.cat(regularization_latents_view1, dim=0)
        all_latents_view2 = (
            torch.cat(regularization_latents_view2, dim=0)
            if regularization_latents_view2 is not None
            else None
        )
        all_original = (
            torch.cat(regularization_original, dim=0)
            if regularization_original is not None
            else None
        )
        same_sequence_mask = None
        if self.CSE_W != 0.0 or self.LG_W != 0.0:
            all_regularization_input_ids = torch.cat(regularization_input_ids, dim=0)
            all_regularization_attention_masks = torch.cat(
                regularization_attention_masks,
                dim=0,
            )
            same_sequence_mask = (
                all_regularization_input_ids[:, None, :]
                == all_regularization_input_ids[None, :, :]
            ).all(dim=-1) & (
                all_regularization_attention_masks[:, None, :]
                == all_regularization_attention_masks[None, :, :]
            ).all(dim=-1)

        regularization_loss = self.compute_unified_regularization_loss(
            all_latents_view1,
            latents_original=all_original,
            latents_view2=all_latents_view2,
            same_sequence_mask=same_sequence_mask,
        )

        total_loss = supervised_loss + regularization_loss
        return ModelOutput(
            loss=total_loss,
            logits=logits,
            hidden_state_virus=virus_regressor_out,
        )


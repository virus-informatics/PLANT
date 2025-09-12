

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, EsmConfig, PreTrainedModel
from transformers.utils import ModelOutput
from safetensors.torch import load_file as safe_load

OHE_virus = None
OHE_ref = None
OHE_vp = None
OHE_rp = None

def set_encoders(ohe_virus, ohe_ref, ohe_vp, ohe_rp):
    global OHE_virus, OHE_ref, OHE_vp, OHE_rp
    OHE_virus, OHE_ref, OHE_vp, OHE_rp = ohe_virus, ohe_ref, ohe_vp, ohe_rp


class PatchedsemanticESM(PreTrainedModel):
    config_class = EsmConfig

    def __init__(
        self,
        config,
        esm_model_name,
        effects_len: Optional[int] = None,
        virus_effects_len: Optional[int] = None,
        embed_scale_factor: float = 1,
        latent_dim: int = 3,
        intermediate_dim: int = 256,
        intermediate_dim_encoder: int = 64,
        dropout: float = 0.05,
        dropout_encoder: float = 0.1,
        MAIN_W: float = 1,
        CSE_W: float = 0,
        CSE_ALPHA: float = 0,
        SEMANTIC_W: float = 0.2,
        CSE_W_VIRUS_ONLY: float = 0,
        SEMANTIC_W_VIRUS_ONLY: float = 0.2,
        CART_W: float = 0.05,
        LG_W: float = 0.01
    ):
        super().__init__(config)

        if virus_effects_len is None:
            virus_effects_len = len(OHE_virus.categories_[0]) if OHE_virus is not None else 0
        if effects_len is None:
            e_ref = len(OHE_ref.categories_[0]) if OHE_ref is not None else 0
            e_vp  = len(OHE_vp.categories_[0]) if OHE_vp is not None else 0
            e_rp  = len(OHE_rp.categories_[0]) if OHE_rp is not None else 0
            effects_len = e_ref + e_vp + e_rp

        self.esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        self.esm_model_original = self._initialize_frozen_esm_model(esm_model_name)

        self.embedding_dim = self.esm_model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, latent_dim),
        )

        self.virus_effects = nn.Linear(virus_effects_len, 1, bias=False) if virus_effects_len > 0 else None
        self.systematic_error_effects = (
            nn.Sequential(
                nn.Linear(effects_len, intermediate_dim_encoder),
                nn.ReLU(),
                nn.Dropout(dropout_encoder),
                nn.Linear(intermediate_dim_encoder, 1, bias=False),
            ) if effects_len > 0 else None
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
        self.embed_scale_factor = embed_scale_factor

    def _initialize_frozen_esm_model(self, esm_model_name):
        esm_model = AutoModel.from_pretrained(esm_model_name, add_pooling_layer=False)
        for param in esm_model.parameters():
            param.requires_grad = False
        return esm_model

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_sequence(self, model, input_ids, attention_mask):
        encoder_out = model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return encoder_out.last_hidden_state[:, 0, :]

    def encode_one_hot(self, encoder, input_tensor):
        if encoder is None or input_tensor is None:
            return None
        model_dtype = next(self.parameters()).dtype
        return torch.tensor(
            encoder.transform(input_tensor.detach().cpu().numpy().reshape(-1, 1)).toarray(),
            dtype=model_dtype,
        ).to(self.device)

    def extract_pooled_embeddings(self, model, input_ids, attention_mask):
        with torch.no_grad():
            encoder_out = model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).last_hidden_state

        masked_sum = (encoder_out * attention_mask.to(self.device).unsqueeze(-1)).sum(dim=1)
        mask_count = attention_mask.to(self.device).sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = masked_sum / mask_count

        max_pooled = torch.max(encoder_out, dim=1)[0]
        pooled_embedding = torch.cat([mean_pooled, max_pooled], dim=-1)
        return pooled_embedding

    def compute_semantic_loss(self, latents, latents_original, embed_scale, embed_scale_factor):
        pairwise_distances = torch.cdist(latents, latents, p=2)
        upper_triangle_indices = torch.triu_indices(
            pairwise_distances.size(0), pairwise_distances.size(1), offset=1
        )
        pairwise_distances_original = torch.cdist(latents_original, latents_original, p=2)
        pairwise_distances_original = pairwise_distances_original / embed_scale_factor

        semantic_loss = self.mse_loss_wo_mean(
            pairwise_distances_original[upper_triangle_indices],
            pairwise_distances[upper_triangle_indices] * embed_scale
        ).mean()
        return semantic_loss

    def contrastive_loss_semantic(self, embeddings1, embeddings2, margin=1.0, alpha=1.0):
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)
        positive_loss = torch.mean(torch.diag(distance_matrix))
        margin_loss = torch.clamp(margin - distance_matrix, min=0)
        weight = torch.exp(-alpha * distance_matrix)
        negative_loss = torch.mean(weight * margin_loss)
        return positive_loss + negative_loss

    def local_global_loss(self, latents, k_local=3, margin_global=0.125):
        dists = torch.cdist(latents, latents, p=2)
        N = latents.size(0)
        eye_mask = torch.eye(N, device=dists.device).bool()
        dists_no_self = dists.masked_fill(eye_mask, float('inf'))

        k_safe = min(k_local, N - 1)
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
        predictions,
        predictions_cart,
        targets,
        censors,
        virus_regressor_out,
        virus_regressor_out2,
        reference_regressor_out,
        reference_regressor_out2,
        virus_embedding_original,
        reference_embedding_original,
        weight
    ):
        uncensored_loss = self.mse_loss_wo_mean(predictions, targets) * (1 - censors)
        censored_loss   = self.mse_loss_wo_mean(predictions, torch.minimum(predictions, targets)) * censors
        uncensored_loss = uncensored_loss * weight
        censored_loss   = censored_loss * weight

        uncensored_loss_cart = self.mse_loss_wo_mean(predictions_cart, targets) * (1 - censors)
        censored_loss_cart   = self.mse_loss_wo_mean(predictions_cart, torch.minimum(predictions_cart, targets)) * censors
        uncensored_loss_cart = uncensored_loss_cart * weight
        censored_loss_cart   = censored_loss_cart * weight

        combined_latents = torch.cat([virus_regressor_out, reference_regressor_out], dim=0)
        combined_latents2 = torch.cat([virus_regressor_out2, reference_regressor_out2], dim=0)
        combined_latents_original = torch.cat([virus_embedding_original, reference_embedding_original], dim=0)

        contrastive_loss_value = self.contrastive_loss_semantic(combined_latents, combined_latents2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(combined_latents, k_local=3, margin_global=0.125)
        semantic_loss = self.compute_semantic_loss(combined_latents, combined_latents_original, self.embed_scale, self.embed_scale_factor)

        total_loss = (
            torch.mean(uncensored_loss) * self.MAIN_W
            + torch.mean(censored_loss) * self.MAIN_W
            + torch.mean(uncensored_loss_cart) * self.CART_W
            + torch.mean(censored_loss_cart) * self.CART_W
            + contrastive_loss_value * self.CSE_W
            + semantic_loss * self.SEMANTIC_W
            + local_global_loss_value * self.LG_W
        )
        return total_loss

    def custom_loss_only_semantic(self, virus_regressor_out, virus_regressor_out2, virus_embedding_original):
        contrastive_loss_value = self.contrastive_loss_semantic(virus_regressor_out, virus_regressor_out2, alpha=self.CSE_ALPHA)
        local_global_loss_value = self.local_global_loss(virus_regressor_out, k_local=3, margin_global=0.125)
        semantic_loss = self.compute_semantic_loss(virus_regressor_out, virus_embedding_original, self.embed_scale, self.embed_scale_factor)
        total_loss = (
            semantic_loss * self.SEMANTIC_W_VIRUS_ONLY
            + contrastive_loss_value * self.CSE_W_VIRUS_ONLY
            + local_global_loss_value * self.LG_W
        )
        return total_loss

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        **kwargs,
    ):
        from transformers import EsmConfig
        from pathlib import Path

        if config is None:
            config = EsmConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, *model_args, **kwargs)

        checkpoint_dir = Path(pretrained_model_name_or_path)
        safetensor_files = sorted(checkpoint_dir.glob("model-*-of-*.safetensors"))
        print(f"[INFO] Found {len(safetensor_files)} safetensors.")

        state_dict = {}
        for f in safetensor_files:
            print(f"[INFO] Loading {f}")
            part = safe_load(str(f), device="cpu")
            state_dict.update(part)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("[INFO] Model state_dict loaded.")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        return model

    def forward(
        self,
        input_ids_virus: torch.Tensor,
        attention_mask_virus: torch.Tensor,
        input_ids_reference: Optional[torch.Tensor] = None,
        attention_mask_reference: Optional[torch.Tensor] = None,
        censors=None,
        labels=None,
        virus=None,
        reference=None,
        dates=None,
        virus_passage=None,
        reference_passage=None,
        weight=None,
        **kwargs,
    ):
        device = self.device

        # --- virus 側の潜在 ---
        virus_regressor_out  = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))
        virus_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_virus, attention_mask_virus))
        virus_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_virus, attention_mask_virus)

        if input_ids_reference is None or attention_mask_reference is None:
            return ModelOutput(loss=torch.tensor(0.0, device=device), logits=None, hidden_state_virus=virus_regressor_out)

        reference_regressor_out  = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
        reference_regressor_out2 = self.regressor(self.encode_sequence(self.esm_model, input_ids_reference, attention_mask_reference))
        reference_embedding_original = self.extract_pooled_embeddings(self.esm_model_original, input_ids_reference, attention_mask_reference)

        distance = torch.norm(virus_regressor_out - reference_regressor_out, p=2, dim=1, keepdim=True)

        def _safe_ohe(encoder, x):
            if encoder is None or x is None:
                return None
            return self.encode_one_hot(encoder, x)

        virus_encoding             = _safe_ohe(OHE_virus, virus)
        reference_encoding         = _safe_ohe(OHE_ref, reference)
        virus_passage_encoding     = _safe_ohe(OHE_vp, virus_passage)
        reference_passage_encoding = _safe_ohe(OHE_rp, reference_passage)

        systematic_error = 0.0
        if self.virus_effects is not None and virus_encoding is not None:
            systematic_error1 = self.virus_effects(virus_encoding)
            systematic_error  = systematic_error + systematic_error1
        else:
            systematic_error1 = torch.zeros_like(distance)

        if self.systematic_error_effects is not None:
            parts = []
            if reference_encoding is not None:         parts.append(reference_encoding)
            if virus_passage_encoding is not None:     parts.append(virus_passage_encoding)
            if reference_passage_encoding is not None: parts.append(reference_passage_encoding)
            if len(parts) > 0:
                combined_encoding = torch.cat(parts, dim=-1)
                systematic_error2 = self.systematic_error_effects(combined_encoding)
                systematic_error  = systematic_error + systematic_error2
            else:
                systematic_error2 = torch.zeros_like(distance)
        else:
            systematic_error2 = torch.zeros_like(distance)

        observed_distance = distance + systematic_error
        logits = torch.cat((observed_distance, distance), dim=1)  # [:,0]=observed, [:,1]=cartography

        if labels is None:
            total_loss = torch.tensor(0.0, device=device)
            return ModelOutput(loss=total_loss, logits=logits, hidden_state_virus=virus_regressor_out)

        labels  = labels.to(device).view(-1, 1)
        censors = (torch.zeros_like(labels) if censors is None else censors.to(device).view(-1, 1))
        weight  = (torch.ones_like(labels)  if weight  is None else weight.to(device).view(-1, 1))

        combined_loss = self.custom_loss(
            observed_distance,
            distance,
            labels,
            censors,
            virus_regressor_out,
            virus_regressor_out2,
            reference_regressor_out,
            reference_regressor_out2,
            virus_embedding_original,
            reference_embedding_original,
            weight
        )

        reg = torch.tensor(0.0, device=device)
        if self.virus_effects is not None:
            reg = reg + torch.mean(systematic_error1 ** 2)
        if self.systematic_error_effects is not None:
            reg = reg + torch.mean(systematic_error2 ** 2)
        combined_loss = combined_loss + reg * 1.0E-4

        total_loss = combined_loss
        return ModelOutput(loss=total_loss, logits=logits, hidden_state_virus=virus_regressor_out)
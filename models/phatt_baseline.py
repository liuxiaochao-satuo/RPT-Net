"""
PH-Att Baseline / RPT-Net unified model.

Supports 4 configurations via config switches:
  - PH-Att Baseline:  UE=off, RA=off
  - PH-Att + UE:      UE=on,  RA=off
  - PH-Att + RA:      UE=off, RA=on
  - RPT-Net (full):   UE=on,  RA=on

Pipeline:
    Skeleton [B,C,T,V,M]
    -> Part Heatmap Generation [B,T,P,H,W] + raw_peaks [B,T,P]
    -> Shared CNN Encoder [B,T,P,D]
    -> (optional) Uncertainty Encoding
    -> Part Embedding + Temporal Embedding
    -> Spatial-Temporal Attention Blocks (with optional Reliability bias)
    -> Global Average Pooling [B,D]
    -> Classifier [B, num_classes]
"""

import torch
import torch.nn as nn
from models.modules.part_heatmap_generator import PartHeatmapGenerator, DEFAULT_PARTS
from models.modules.shared_cnn_encoder import SharedCNNEncoder
from models.modules.st_attention_block import STAttentionBlock
from models.modules.uncertainty_estimator import UncertaintyEstimator
from models.modules.uncertainty_encoder import UncertaintyEncoder
from models.modules.reliability_estimator import ReliabilityEstimator


class PHAttBaseline(nn.Module):

    def __init__(
        self,
        num_classes=60,
        num_frames=64,
        num_parts=5,
        heatmap_size=32,
        sigma=1.5,
        token_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_drop=0.1,
        classifier_drop=0.5,
        parts=None,
        use_uncertainty_encoding=False,
        use_reliability_attention=False,
        use_confidence=True,
        use_dispersion=True,
        use_temporal_instability=True,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_parts = num_parts
        self.token_dim = token_dim
        self.use_uncertainty_encoding = use_uncertainty_encoding
        self.use_reliability_attention = use_reliability_attention
        self.need_uncertainty = use_uncertainty_encoding or use_reliability_attention

        # --- Module 1: Part Heatmap Generator ---
        self.heatmap_generator = PartHeatmapGenerator(
            heatmap_size=heatmap_size, sigma=sigma, parts=parts,
        )

        # --- Module 2: Shared CNN Encoder ---
        self.cnn_encoder = SharedCNNEncoder(
            in_channels=1, channels=(32, 64, 128), out_dim=token_dim,
        )

        # --- Module 3: Uncertainty modules (conditional) ---
        if self.need_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(
                use_confidence=use_confidence,
                use_dispersion=use_dispersion,
                use_temporal_instability=use_temporal_instability,
            )
            uncertainty_dim = self.uncertainty_estimator.num_indicators

            if use_uncertainty_encoding:
                self.uncertainty_encoder = UncertaintyEncoder(
                    token_dim=token_dim, uncertainty_dim=uncertainty_dim,
                )

            if use_reliability_attention:
                self.reliability_estimator = ReliabilityEstimator(
                    uncertainty_dim=uncertainty_dim,
                )

        # --- Module 4: Embeddings ---
        self.part_embedding = nn.Parameter(
            torch.zeros(1, 1, num_parts, token_dim))
        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, token_dim))
        nn.init.trunc_normal_(self.part_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

        # --- Module 5: ST Attention Blocks ---
        self.blocks = nn.ModuleList([
            STAttentionBlock(
                dim=token_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(token_dim)

        # --- Module 6: Classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(classifier_drop),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, x, return_attention=False, apply_reliability_bias=True):
        """
        Args:
            x: [B, C, T, V, M]
            return_attention: if True, return auxiliary dict
            apply_reliability_bias: if False, attention blocks run without
                reliability bias (for Standard vs RA visualization)

        Returns:
            logits: [B, num_classes]
            aux: dict (only if return_attention=True)
        """
        aux = {}

        # Step 1: Generate part heatmaps
        part_heatmaps, raw_peaks = self.heatmap_generator(x)
        B, T, P, H, W = part_heatmaps.shape

        if return_attention:
            aux['part_heatmaps'] = part_heatmaps.detach()

        # Step 2: CNN encoding
        cnn_input = part_heatmaps.reshape(B * T * P, 1, H, W)
        part_tokens = self.cnn_encoder(cnn_input).reshape(B, T, P, -1)

        # Step 3: Uncertainty estimation + encoding (before embeddings)
        uncertainty = None
        reliability = None

        if self.need_uncertainty:
            uncertainty = self.uncertainty_estimator(part_heatmaps, raw_peaks)
            if return_attention:
                aux['uncertainty'] = uncertainty.detach()

            if self.use_uncertainty_encoding:
                part_tokens = self.uncertainty_encoder(part_tokens, uncertainty)

            if self.use_reliability_attention:
                reliability = self.reliability_estimator(uncertainty)
                if return_attention:
                    aux['reliability'] = reliability.detach()

        # Step 4: Add embeddings
        part_tokens = part_tokens + self.part_embedding + self.time_embedding[:, :T]

        # Step 5: Spatial-Temporal Attention
        attn_reliability = reliability
        if not apply_reliability_bias:
            attn_reliability = None

        tokens = part_tokens
        all_attn = []
        for block in self.blocks:
            if return_attention:
                tokens, attn_dict = block(
                    tokens, reliability=attn_reliability, return_attn=True)
                all_attn.append(attn_dict)
            else:
                tokens = block(tokens, reliability=attn_reliability)

        tokens = self.norm(tokens)

        if return_attention:
            aux['attention_maps'] = all_attn
            aux['part_tokens'] = tokens.detach()

        # Step 6: Global Average Pooling + Classification
        feat = tokens.mean(dim=(1, 2))
        logits = self.classifier(feat)

        if return_attention:
            return logits, aux
        return logits


def build_phatt_baseline(cfg):
    """Build PH-Att Baseline / RPT-Net from config dict."""
    parts = None
    if 'parts' in cfg and 'layout' in cfg['parts']:
        parts = cfg['parts']['layout']

    model = PHAttBaseline(
        num_classes=cfg.get('num_classes', 60),
        num_frames=cfg.get('num_frames', 64),
        num_parts=cfg.get('num_parts', 5),
        heatmap_size=cfg.get('heatmap_size', 32),
        sigma=cfg.get('sigma', 1.5),
        token_dim=cfg.get('token_dim', 128),
        depth=cfg.get('depth', 4),
        num_heads=cfg.get('num_heads', 4),
        mlp_ratio=cfg.get('mlp_ratio', 4.0),
        dropout=cfg.get('dropout', 0.1),
        attn_drop=cfg.get('attn_drop', 0.1),
        classifier_drop=cfg.get('classifier_drop', 0.5),
        parts=parts,
        use_uncertainty_encoding=cfg.get('use_uncertainty_encoding', False),
        use_reliability_attention=cfg.get('use_reliability_attention', False),
        use_confidence=cfg.get('use_confidence', True),
        use_dispersion=cfg.get('use_dispersion', True),
        use_temporal_instability=cfg.get('use_temporal_instability', True),
    )
    return model

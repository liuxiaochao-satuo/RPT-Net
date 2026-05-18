"""
PH-Att Baseline: Part Heatmap Attention Baseline for skeleton action recognition.

Pipeline:
    Skeleton [B,C,T,V,M]
    → Part Heatmap Generation [B,T,P,H,W]
    → Shared CNN Encoder [B,T,P,D]
    → Part Embedding + Temporal Embedding
    → Spatial-Temporal Attention Blocks x depth
    → Global Average Pooling [B,D]
    → Classifier [B, num_classes]
"""

import torch
import torch.nn as nn
from models.modules.part_heatmap_generator import PartHeatmapGenerator, DEFAULT_PARTS
from models.modules.shared_cnn_encoder import SharedCNNEncoder
from models.modules.st_attention_block import STAttentionBlock


class PHAttBaseline(nn.Module):
    """Part Heatmap Attention Baseline.

    Args:
        num_classes: number of action classes
        num_frames: number of input frames (T)
        num_parts: number of body parts (P)
        heatmap_size: spatial size of part heatmaps (H=W)
        sigma: Gaussian kernel std for heatmap generation
        token_dim: dimension of part tokens (D)
        depth: number of ST attention blocks
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: dropout rate
        attn_drop: attention dropout rate
        classifier_drop: classifier dropout rate
        parts: dict of body part definitions
        use_uncertainty_encoding: placeholder for UE module
        use_reliability_attention: placeholder for RA module
    """

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
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_parts = num_parts
        self.token_dim = token_dim
        self.use_uncertainty_encoding = use_uncertainty_encoding
        self.use_reliability_attention = use_reliability_attention

        # --- Module 1: Part Heatmap Generator ---
        self.heatmap_generator = PartHeatmapGenerator(
            heatmap_size=heatmap_size,
            sigma=sigma,
            parts=parts,
        )

        # --- Module 2: Shared CNN Encoder ---
        self.cnn_encoder = SharedCNNEncoder(
            in_channels=1,
            channels=(32, 64, 128),
            out_dim=token_dim,
        )

        # --- Module 3: Embeddings ---
        self.part_embedding = nn.Parameter(
            torch.zeros(1, 1, num_parts, token_dim))
        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, token_dim))
        nn.init.trunc_normal_(self.part_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

        # --- Module 4: ST Attention Blocks ---
        self.blocks = nn.ModuleList([
            STAttentionBlock(
                dim=token_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(token_dim)

        # --- Module 5: Classifier ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Dropout(classifier_drop),
            nn.Linear(token_dim, num_classes),
        )

        # --- Placeholder: Uncertainty Encoding ---
        # if self.use_uncertainty_encoding:
        #     self.uncertainty_encoder = UncertaintyEncoder(...)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, C, T, V, M]
            return_attention: if True, return auxiliary dict with attention maps

        Returns:
            logits: [B, num_classes]
            aux: dict (only if return_attention=True)
        """
        aux = {}

        # Step 1: Generate part heatmaps
        part_heatmaps = self.heatmap_generator(x)  # [B, T, P, H, W]
        if return_attention:
            aux['part_heatmaps'] = part_heatmaps.detach()

        B, T, P, H, W = part_heatmaps.shape

        # Step 2: CNN encoding (shared across all parts and frames)
        cnn_input = part_heatmaps.reshape(B * T * P, 1, H, W)
        part_tokens = self.cnn_encoder(cnn_input)  # [B*T*P, D]
        part_tokens = part_tokens.reshape(B, T, P, -1)  # [B, T, P, D]

        # Step 3: Add embeddings
        part_tokens = part_tokens + self.part_embedding + self.time_embedding[:, :T]

        # Placeholder: Uncertainty Encoding
        reliability = None
        # if self.use_uncertainty_encoding:
        #     uncertainty = self.compute_uncertainty(part_heatmaps)
        #     part_tokens = self.uncertainty_encoder(part_tokens, uncertainty)
        #     if self.use_reliability_attention:
        #         reliability = self.compute_reliability(uncertainty)

        # Step 4: Spatial-Temporal Attention
        tokens = part_tokens
        all_attn = []
        for block in self.blocks:
            if return_attention:
                tokens, attn_dict = block(
                    tokens, reliability=reliability, return_attn=True)
                all_attn.append(attn_dict)
            else:
                tokens = block(tokens, reliability=reliability)

        tokens = self.norm(tokens)

        if return_attention:
            aux['attention_maps'] = all_attn
            aux['part_tokens'] = tokens.detach()

        # Step 5: Global Average Pooling + Classification
        feat = tokens.mean(dim=(1, 2))  # [B, D]
        logits = self.classifier(feat)  # [B, num_classes]

        if return_attention:
            return logits, aux
        return logits


def build_phatt_baseline(cfg):
    """Build PH-Att Baseline from config dict."""
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
    )
    return model

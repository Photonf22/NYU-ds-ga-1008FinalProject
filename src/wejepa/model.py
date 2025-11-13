"""Simplified I-JEPA style encoder used throughout the wejepa package."""
from __future__ import annotations

import copy
import math
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed(nn.Module):
    """Turn an image into non-overlapping patch tokens."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = rearrange(x, "b e h w -> b (h w) e")
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Predictor(nn.Module):
    """Decoder that transforms context tokens into target predictions."""

    def __init__(self, embed_dim: int, num_heads: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_encoding: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        x = torch.cat((context_encoding, target_masks), dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        target_tokens = target_masks.shape[1]
        return x[:, -target_tokens:, :]


class IJEPA_base(nn.Module):
    """Simplified implementation of the Image-based JEPA architecture."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        enc_depth: int,
        pred_depth: int,
        num_heads: int,
        post_emb_norm: bool = False,
        M: int = 4,
        mode: str = "train",
        layer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del layer_dropout  # kept for backwards compatibility
        self.M = M
        self.mode = mode
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_dim = self.patch_embed.patch_shape
        self.num_tokens = self.patch_dim[0] * self.patch_dim[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.student_encoder = TransformerEncoder(embed_dim, enc_depth, num_heads)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        self.predictor = Predictor(embed_dim, num_heads, pred_depth)

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    @torch.no_grad()
    def momentum_update(self, momentum: float) -> None:
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(), self.teacher_encoder.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)

    @torch.no_grad()
    def get_target_block(
        self,
        target_encoder: nn.Module,
        x: torch.Tensor,
        patch_dim: Tuple[int, int],
        aspect_ratio: float,
        scale: float,
        M: int,
    ) -> Tuple[torch.Tensor, List[List[int]], List[int]]:
        target_encoder = target_encoder.eval()
        enc = target_encoder(x)
        enc = self.norm(enc)
        patch_h, patch_w = patch_dim
        num_tokens = patch_h * patch_w
        num_patches_block = max(1, int(num_tokens * scale))
        block_h = max(1, int(round(math.sqrt(num_patches_block / max(aspect_ratio, 1e-6)))))
        block_w = max(1, int(round(block_h * aspect_ratio)))
        block_h = min(block_h, patch_h)
        block_w = min(block_w, patch_w)
        device = x.device
        target_block = torch.zeros(
            (M, x.shape[0], block_h * block_w, x.shape[2]), device=device, dtype=x.dtype
        )
        target_patches: List[List[int]] = []
        all_patches: List[int] = []
        for block_idx in range(M):
            start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w
            patches: List[int] = []
            for i in range(block_h):
                for j in range(block_w):
                    idx = start_patch + i * patch_w + j
                    patches.append(idx)
                    if idx not in all_patches:
                        all_patches.append(idx)
            target_patches.append(patches)
            target_block[block_idx] = enc[:, patches, :]
        return target_block, target_patches, all_patches

    def get_context_block(
        self,
        x: torch.Tensor,
        patch_dim: Tuple[int, int],
        aspect_ratio: float,
        scale: float,
        target_patches: Sequence[int],
    ) -> torch.Tensor:
        patch_h, patch_w = patch_dim
        num_tokens = patch_h * patch_w
        num_patches_block = max(1, int(num_tokens * scale))
        block_h = max(1, int(round(math.sqrt(num_patches_block / max(aspect_ratio, 1e-6)))))
        block_w = max(1, int(round(block_h * aspect_ratio)))
        block_h = min(block_h, patch_h)
        block_w = min(block_w, patch_w)
        start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        patches: List[int] = []
        target_set = set(target_patches)
        for i in range(block_h):
            for j in range(block_w):
                idx = start_patch + i * patch_w + j
                if idx not in target_set:
                    patches.append(idx)
        if not patches:
            available = [idx for idx in range(num_tokens) if idx not in target_set]
            patches = available if available else list(range(num_tokens))
        return x[:, patches, :]

    def forward(
        self,
        x: torch.Tensor,
        target_aspect_ratio: float = 1.0,
        target_scale: float = 0.2,
        context_aspect_ratio: float = 1.0,
        context_scale: float = 0.9,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embedding
        tokens = self.post_emb_norm(tokens)
        if self.mode == "test":
            return self.student_encoder(tokens)
        target_blocks, target_patches, all_patches = self.get_target_block(
            self.teacher_encoder, tokens, self.patch_dim, target_aspect_ratio, target_scale, self.M
        )
        context_block = self.get_context_block(
            tokens, self.patch_dim, context_aspect_ratio, context_scale, all_patches
        )
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)
        m, bsz, n_tok, embed_dim = target_blocks.shape
        prediction_blocks = torch.zeros_like(target_blocks)
        for i in range(m):
            block_len = len(target_patches[i])
            target_masks = self.mask_token.repeat(bsz, block_len, 1)
            pos_embed = self.pos_embedding[:, target_patches[i], :]
            target_masks = target_masks + pos_embed
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)
        return prediction_blocks, target_blocks


__all__ = ["IJEPA_base", "PatchEmbed"]

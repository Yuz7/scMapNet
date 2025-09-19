# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.jit import Final
from timm.layers import DropPath, Mlp

import timm.models.vision_transformer

#attention定义，主要是qkv计算
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        weight = attn
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weight

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# block定义，其中包括了attention , mlp , norm，数据会经过这个block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, weights = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(h))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, weights

# 这个ViT继承了timm中的ViT，有些调用是从父对象中调用的，需要看一下父对象的方法
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, rollout_map=True, **kwargs):
        super(VisionTransformer, self).__init__(block_fn=Block, **kwargs)

        self.global_pool = global_pool
        self.rollout_map = rollout_map
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # print(f'attn test for x shape 0:{x.shape}')
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # print(f'attn test for x shape 1:{x.shape}')
        attn_weights = []
        for blk in self.blocks:
            # x = blk(x)
            x, weights = blk(x)
            # print(f'attn test for weights shape:{weights.shape}')
            attn_weights.append(weights)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            # print(f'attn test for outcome shape:{outcome.shape}')
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, get_weight(attn_weights)
    
    def forward(self, x):
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        # print(f'attn test for outcome shape:{x.shape}')
        return x, attn_weights

# 这个get_weight实现了 attention rollout，将attention分数整合提取出来，用于基因重要性提取和模型解释性的探究
def get_weight(att_mat):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)
    # print(att_mat.size())
    # Average the attention weights across all heads.
    if len(att_mat.size()) <= 4:
        att_mat = torch.mean(att_mat, dim=1)
    else:
        att_mat = torch.mean(att_mat, dim=2)
    # print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    if len(att_mat.size()) < 4:
        print(att_mat.size())
        residual_att = torch.eye(att_mat.size(2))
    else:
        # print(att_mat.size())
        residual_att = torch.eye(att_mat.size(3))
    # print(residual_att.size())
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    if len(v.size()) < 3:
        v = v[0,1:]
    else:
        v = v[:,0,1:]
    #print(v.size())
    return v

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

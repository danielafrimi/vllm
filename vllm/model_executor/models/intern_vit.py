# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from collections import namedtuple
from collections.abc import Iterable
from functools import partial
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from itertools import repeat
import collections.abc


from vllm.model_executor.models.input_conditionerV1 import InputConditioner, get_default_conditioner
from vllm.model_executor.models.vit_patch_generator_v1 import ViTPatchGenerator

NORM2FN = {
    "rms_norm": RMSNorm,
    "layer_norm": nn.LayerNorm,
}

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple



def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    # if not torch.jit.is_scripting() and verbose:
    #     _logger.info(f'Resized position embedding: {old_size} to {new_size}.')

    return posemb

class InternVisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size

        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int):
        target_dtype = pos_embed.dtype
        pos_embed = (pos_embed.float().reshape(
            1,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            -1,
        ).permute(0, 3, 1, 2))
        pos_embed = F.interpolate(pos_embed,
                                  size=(H, W),
                                  mode="bicubic",
                                  align_corners=False)
        return pos_embed.reshape(1, -1, H * W).permute(0, 2,
                                                       1).to(target_dtype)

    def _get_position_embedding(self, H: int, W: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if self.num_patches == H * W:
            return position_embedding

        return torch.cat(
            [
                position_embedding[:, :1, :],
                self._get_pos_embed(position_embedding[:, 1:, :], H, W),
            ],
            dim=1,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            target_dtype))  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1,
                                                   -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    # output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            config: PretrainedConfig,
            # img_size: Union[int, Tuple[int, int]] = 224,
            # patch_size: int = 16,
            # in_chans: int = 3,
            # embed_dim: int = 768,
            # norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            # output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True, # todo where do we pass this in config ??
            no_embed_class: bool = False, # todo where do we pass this in config ??
            dynamic_img_pad: bool = False, # todo where do we pass this in config ??
            pos_embed: str = 'learn', # todo where do we pass this in config ??
    ):
        super().__init__()
        self.patch_size = to_2tuple(config.patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(config.image_size)

        # if output_fmt is not None:
        #     self.flatten = False
        #     self.output_fmt = Format(output_fmt)
        # else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        # self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.reg_tokens = nn.Parameter(torch.zeros(1, config.reg_tokens, config.hidden_size)) if config.reg_tokens else None
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens = 1 + config.reg_tokens

        embed_len = self.num_patches if no_embed_class else self.num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, config.hidden_size) * .02)
        

        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size, bias=bias)
        self.norm = NORM2FN[config.norm_type](config.hidden_size) if config.norm_type else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    # def set_input_size(
    #         self,
    #         img_size: Optional[Union[int, Tuple[int, int]]] = None,
    #         patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    # ):
    #     new_patch_size = None
    #     if patch_size is not None:
    #         new_patch_size = to_2tuple(patch_size)
    #     if new_patch_size is not None and new_patch_size != self.patch_size:
    #         with torch.no_grad():
    #             new_proj = nn.Conv2d(
    #                 self.proj.in_channels,
    #                 self.proj.out_channels,
    #                 kernel_size=new_patch_size,
    #                 stride=new_patch_size,
    #                 bias=self.proj.bias is not None,
    #             )
    #             new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
    #             if self.proj.bias is not None:
    #                 new_proj.bias.copy_(self.proj.bias)
    #             self.proj = new_proj
    #         self.patch_size = new_patch_size
    #     img_size = img_size or self.img_size
    #     if img_size != self.img_size or new_patch_size is not None:
    #         self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        # if self.flatten:
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        # elif self.output_fmt != Format.NCHW:
        #     x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        x = self._pos_embed(x)
        return x
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input."""
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_tokens is not None:
            to_cat.append(self.reg_tokens.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return x

class InternVisionPatchModel(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                "You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f"wrong pixel_values size: {pixel_values.shape}")

        return hidden_states


class InternParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.num_heads_per_partition = divide(num_dummy_heads + self.num_heads,
                                              self.tp_size)

        self.scale = self.head_dim**-0.5
        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
            self.k_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )

        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        out = self.attn(q, k, v)
        out, _ = self.proj(out)
        return out


class InternSdpaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        num_dummy_heads: int = 0,
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}).")

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim,
                             3 * self.dummy_dim,
                             bias=config.qkv_bias)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )
            self.k_norm = RMSNorm(
                self.dummy_dim,
                eps=config.layer_norm_eps,
                var_hidden_size=self.embed_dim,
            )

        self.proj = nn.Linear(self.dummy_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        if self.qk_normalization:
            B_, N_, H_, D_ = q.shape
            q = self.q_norm(q.flatten(-2, -1)).view(B_, N_, H_, D_)
            k = self.k_norm(k.flatten(-2, -1)).view(B_, N_, H_, D_)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        return x


class InternMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = self._init_attn(
            config,
            quant_config,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.attn",
        )

        self.mlp = InternMLP(config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp")
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))

    def _init_attn(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        num_dummy_heads: int,
        prefix: str = "",
    ):
        # fallback to sdpa attention if tp unavailable
        tp_size = get_tensor_model_parallel_world_size()
        num_heads = config.num_attention_heads

        if (num_heads + num_dummy_heads) % tp_size == 0:
            return InternParallelAttention(
                config,
                quant_config=quant_config,
                num_dummy_heads=num_dummy_heads,
                prefix=prefix,
            )

        return InternSdpaAttention(config, num_dummy_heads=num_dummy_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = (hidden_states +
                         self.attn(self.norm1(hidden_states)) * self.ls1)

        hidden_states = (hidden_states +
                         self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states


class InternVisionEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ):
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(
                config,
                quant_config,
                num_dummy_heads=num_dummy_heads,
                prefix=f"{prefix}.layers.{layer_idx}",
            ) for layer_idx in range(num_hidden_layers)
        ])

    def forward(self, inputs_embeds: torch.Tensor):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class InternVisionModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        # self.embeddings = InternVisionEmbeddings(config)
        self.embeddings = PatchEmbed(config)
        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
        )

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                "You have to specify pixel_values or pixel_embeds")

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f"wrong pixel_values size: {pixel_values.shape}")

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)

        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params






class InternVisionModelModified(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig = None,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(to_2tuple(config.patch_size), config.image_size)
        max_img_size = int(round(config.max_img_size / config.patch_size) * config.patch_size)
        self.patch_generator = ViTPatchGenerator(config.patch_size, config.hidden_size, input_dims=self.img_size, max_input_dims=max_img_size, cls_token=True, register_multiple=config.args["register_multiple"])

    

        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
        )
    
    def _init_img_size(self, patch_size, img_size: Union[int, Tuple[int, int]]):
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
            x: torch.Tensor
    ) -> torch.FloatTensor:
        assert self.patch_generator is not None
        hidden_states = self.patch_generator(x)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        # todo do we need to add norm
        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params



class RadioModel(nn.Module):
    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)

        self.config = config
        self.input_conditioner = get_default_conditioner()
        self.model = InternVisionModelModified(config=config, quant_config=quant_config, num_hidden_layers_override=num_hidden_layers_override, num_dummy_heads=num_dummy_heads, prefix=prefix)
        
      

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        self.model.to(dtype=dtype)
        self.input_conditioner.dtype = dtype


    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        pixel_values = self.input_conditioner(pixel_values)
        # todo maybe extract final ????????? at the end... 
        return self.model(pixel_values)
        

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params



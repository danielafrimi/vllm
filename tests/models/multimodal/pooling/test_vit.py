# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
# from ....conftest import ImageTestAssets
from PIL import Image
from transformers import (AutoConfig, AutoModel, CLIPImageProcessor,
                          PretrainedConfig)

from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.model_executor.models.intern_vit import RadioModel

# we use snapshot_download to prevent conflicts between
# dynamic_module and trust_remote_code for hf_runner
DOWNLOAD_PATTERN = ["*.json", "*.py", "*.safetensors", "*.txt", "*.model"]


def build_intern_config_from_radio(radio_cfg):
    # Map common ViT names to dims
    vit_dims = {
        "vit_small_patch16_224": (384, 12, 6, 1536),
        "vit_base_patch16_224": (768, 12, 12, 3072),
        "vit_large_patch16_224": (1024, 24, 16, 4096),
        "vit_huge_patch16_224": (1280, 32, 16, 5120),
    }
    model_name = radio_cfg.args["model"]
    hidden_size, num_layers, num_heads, intermediate_size = vit_dims.get(
        model_name,
        (768, 12, 12, 3072)  # safe default (ViT-B/16)
    )

    # Image/patch sizes
    pref_res = getattr(radio_cfg, "preferred_resolution", None)
    image_size = (pref_res[0] if pref_res else 224)
    patch_size = getattr(radio_cfg, "patch_size", 16) or 16

    intern_cfg = PretrainedConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        image_size=image_size,
        patch_size=patch_size,
        qkv_bias=True,
        qk_normalization=False,
        norm_type="layer_norm",
        layer_norm_eps=1e-6,
        initializer_factor=1.0,
        hidden_act="gelu",
        reg_tokens=radio_cfg.args["register_multiple"],
    )
    return intern_cfg


def map_hf_radio_to_vllm_intern(hf_sd: dict, radio_vllm) -> dict:
    mapped = {}
    for k, v in hf_sd.items():
        print(f"key in state dict: {k}")
        if not k.startswith("radio_model."):
            continue
        k2 = k[len("radio_model."):]

        # skip buffers not used in vLLM
        if k2 in {"summary_idxs"}:
            continue

        # patch generator: keep same after stripping prefix
        if k2.startswith("model.patch_generator."):
            mapped_key = f"model.patch_generator.{k2.split('.', 2)[-1]}"
            mapped[mapped_key] = v
            continue

        # input conditioner
        if k2.startswith("input_conditioner."):
            mapped_key = f"input_conditioner.{k2.split('.', 1)[-1]}"
            mapped[mapped_key] = v
            continue

        # blocks -> encoder.layers
        if k2.startswith("model.blocks."):
            parts = k2.split(".")
            layer_idx = parts[2]
            suffix = ".".join(
                parts[3:]
            )  # e.g. norm1.weight, attn.qkv.weight, mlp.fc1.weight, etc.
            # ls1/ls2 do not exist in HF (Identity); vLLM has params – leave them default
            if suffix in {"ls1", "ls2"} or suffix.startswith(("ls1.", "ls2.")):
                continue
            mapped_key = f"model.encoder.layers.{layer_idx}.{suffix}"
            mapped[mapped_key] = v
            continue

    return mapped


VIT_DIMS = {
    "vit_small_patch16_224": (384, 12, 6, 1536),
    "vit_base_patch16_224": (768, 12, 12, 3072),
    "vit_large_patch16_224": (1024, 24, 16, 4096),
    "vit_huge_patch16_224": (1280, 32, 16, 5120),
}


def get_args_from_model_type(model_type):
    return VIT_DIMS[model_type]


@torch.inference_mode()
def _test_radio_vllm_vs_hf():
    hf_repo = "nvidia/C-RADIOv2-H"

    # Init single-process distributed + model parallel so InternVisionModel can construct
    import tempfile
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend=backend,
    )
    initialize_model_parallel(1, 1)

    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
    config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)

    hf_model = AutoModel.from_pretrained(hf_repo,
                                         config=config,
                                         trust_remote_code=True)
    hf_model.eval().cuda()

    images = [
        Image.open('/home/dafrimi/projects/vllm/images/horse.jpg').convert(
            'RGB')
    ]

    pixel_values = [
        image_processor(images, return_tensors='pt').pixel_values.to(
            hf_model.dtype)[:, :, :432, :640] for images in images
    ]

    a = hf_model(pixel_values[0].to("cuda"))

    # hf_outputs_per_image = [
    #     hf_model(pixel_value.to("cuda")).features
    #     for pixel_value in pixel_values
    # ]

    # intern_cfg = build_intern_config_from_radio(config)
    try:
        # intern_vit = InternVisionModel(intern_cfg).to("cuda")
        hidden_size, num_layers, num_heads, intermediate_size = get_args_from_model_type(
            config.args["model"])
        config.num_hidden_layers = num_layers
        config.hidden_size = hidden_size
        config.num_attention_heads = num_heads
        config.intermediate_size = intermediate_size
        config.norm_type = "layer_norm"
        config.image_size = 224
        config.hidden_act = "gelu"
        config.layer_norm_eps = 1e-6
        config.initializer_factor = 1.0
        config.qkv_bias = True
        config.qk_normalization = False
        config.max_img_size = 2048

        radio_vllm = RadioModel(config).to("cuda")

        hf_state_dict = hf_model.state_dict()
        vllm_state_dict = map_hf_radio_to_vllm_intern(hf_state_dict,
                                                      radio_vllm)
        missing, unexpected = radio_vllm.load_state_dict(vllm_state_dict,
                                                         strict=False)
        print(f"missing: {missing}")
        print(f"unexpected: {unexpected}")

        vllm_outputs_per_image = [
            radio_vllm(pixel_values=pixel_value.to("cuda"))
            for pixel_value in pixel_values
        ]

        cos_similar = nn.CosineSimilarity(dim=-1)
        for vllm_output, timm_output in zip(vllm_outputs_per_image, [a]):
            assert cos_similar(vllm_output, timm_output).mean() > 0.99
            print("PASSED")
    finally:
        cleanup_dist_env_and_memory()


# @torch.inference_mode()
# def run_vit_vs_intern_vit(
#     image_assets: ImageTestAssets,
#     model_id: str,
#     *,
#     dtype: str,
# ):
#     model_path = snapshot_download(model_id, allow_patterns=DOWNLOAD_PATTERN)
#     torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

#     img_processor = CLIPImageProcessor.from_pretrained(model_path)
#     images = [asset.pil_image for asset in image_assets]
#     pixel_values = [
#         img_processor(images, return_tensors='pt').pixel_values.to(torch_dtype)
#         for images in images
#     ]

#     config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#     if not getattr(config, "norm_type", None):
#         config.norm_type = "rms_norm"

#     hf_model = AutoModel.from_pretrained(model_path,
#                                          torch_dtype=torch_dtype,
#                                          trust_remote_code=True).to("cuda")

#     vllm_model = InternVisionModel(config)
#     vllm_model.load_weights(hf_model.state_dict().items())

#     del hf_model
#     cleanup_dist_env_and_memory()

#     vllm_model = vllm_model.to("cuda", torch_dtype)
#     vllm_outputs_per_image = [
#         vllm_model(pixel_values=pixel_value.to("cuda"))
#         for pixel_value in pixel_values
#     ]

#     # Build a matching TIMM VisionTransformer and load mapped weights
#     def build_timm_vit_from_config(cfg) -> VisionTransformer:
#         mlp_ratio = cfg.intermediate_size / cfg.hidden_size
#         norm_layer = partial(nn.LayerNorm, eps=cfg.layer_norm_eps)
#         try:
#             if getattr(cfg, "norm_type", None) == "rms_norm":
#                 from timm.layers import RMSNorm as TimmRMSNorm  # type: ignore
#                 norm_layer = partial(TimmRMSNorm, eps=cfg.layer_norm_eps)
#         except Exception:
#             pass
#         init_values = getattr(cfg, "initializer_factor", None)
#         if init_values is not None and isinstance(init_values, (int, float)):
#             init_values = float(init_values)
#         else:
#             init_values = None
#         model = VisionTransformer(
#             img_size=cfg.image_size,
#             patch_size=cfg.patch_size,
#             in_chans=3,
#             num_classes=0,
#             embed_dim=cfg.hidden_size,
#             depth=cfg.num_hidden_layers,
#             num_heads=cfg.num_attention_heads,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=getattr(cfg, "qkv_bias", True),
#             init_values=init_values,
#             norm_layer=norm_layer,
#             drop_rate=0.0,
#             attn_drop_rate=0.0,
#             drop_path_rate=0.0,
#         )
#         return model

#     def map_vllm_to_timm_state_dict(vllm_sd: dict) -> dict:
#         timm_sd = {}
#         for k, v in vllm_sd.items():
#             print(f"k is {k} !!!!")
#             # Embeddings
#             if k == "embeddings.class_embedding":
#                 timm_sd["cls_token"] = v
#                 continue
#             if k == "embeddings.position_embedding":
#                 timm_sd["pos_embed"] = v
#                 continue
#             if k.startswith("embeddings.patch_embedding."):
#                 timm_sd[k.replace("embeddings.patch_embedding",
#                                   "patch_embed.proj")] = v
#                 continue
#             # Encoder blocks
#             if k.startswith("encoder.layers."):
#                 parts = k.split(".")
#                 layer_idx = parts[2]
#                 suffix = ".".join(parts[3:])
#                 if suffix.startswith("attn.qkv."):
#                     timm_sd[f"blocks.{layer_idx}.attn.qkv.{suffix.split('.', 2)[-1]}"] = v
#                     continue
#                 if suffix.startswith("attn.proj."):
#                     timm_sd[f"blocks.{layer_idx}.attn.proj.{suffix.split('.', 2)[-1]}"] = v
#                     continue
#                 if suffix.startswith("norm1."):
#                     timm_sd[f"blocks.{layer_idx}.norm1.{suffix.split('.', 1)[-1]}"] = v
#                     continue
#                 if suffix.startswith("norm2."):
#                     timm_sd[f"blocks.{layer_idx}.norm2.{suffix.split('.', 1)[-1]}"] = v
#                     continue
#                 if suffix.startswith("mlp.fc1."):
#                     timm_sd[f"blocks.{layer_idx}.mlp.fc1.{suffix.split('.', 2)[-1]}"] = v
#                     continue
#                 if suffix.startswith("mlp.fc2."):
#                     timm_sd[f"blocks.{layer_idx}.mlp.fc2.{suffix.split('.', 2)[-1]}"] = v
#                     continue
#                 if suffix == "ls1":
#                     timm_sd[f"blocks.{layer_idx}.ls1.gamma"] = v
#                     continue
#                 if suffix == "ls2":
#                     timm_sd[f"blocks.{layer_idx}.ls2.gamma"] = v
#                     continue
#         return timm_sd

#     timm_model = build_timm_vit_from_config(config).to("cuda", torch_dtype)
#     vllm_sd = {k: v for k, v in vllm_model.state_dict().items()}
#     timm_sd = map_vllm_to_timm_state_dict(vllm_sd)
#     missing, unexpected = timm_model.load_state_dict(timm_sd, strict=False)
#     assert not unexpected, f"Unexpected TIMM keys: {unexpected}"

#     def timm_tokens(model: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
#         B = x.shape[0]
#         x = model.patch_embed(x)
#         if model.cls_token is not None:
#             cls_tokens = model.cls_token.expand(B, -1, -1)
#             x = torch.cat((cls_tokens, x), dim=1)
#         # positional embedding (interpolate grid-only when needed)
#         if x.shape[1] == model.pos_embed.shape[1]:
#             x = x + model.pos_embed
#         else:
#             pos = model.pos_embed
#             cls_pos, grid_pos = pos[:, :1], pos[:, 1:]
#             n = x.shape[1] - 1
#             h = w = int(n ** 0.5)
#             gh = gw = int(grid_pos.shape[1] ** 0.5)
#             grid_pos = grid_pos.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
#             grid_pos = torch.nn.functional.interpolate(
#                 grid_pos, size=(h, w), mode="bicubic", align_corners=False
#             )
#             grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
#             x = x + torch.cat([cls_pos, grid_pos], dim=1)
#         x = model.pos_drop(x)
#         for blk in model.blocks:
#             x = blk(x)
#         return x  # skip final norm to match vLLM output

#     timm_outputs_per_image = [
#         timm_tokens(timm_model, pixel_value.to("cuda"))
#         for pixel_value in pixel_values
#     ]

#     cos_similar = nn.CosineSimilarity(dim=-1)
#     for vllm_output, timm_output in zip(vllm_outputs_per_image,
#                                         timm_outputs_per_image):
#         assert cos_similar(vllm_output, timm_output).mean() > 0.99

# @pytest.mark.parametrize("model_id", [
#     "OpenGVLab/InternViT-300M-448px",
# ])
# @pytest.mark.parametrize("dtype", ["half"])
# def test_vit_vs_intern_vit(dist_init, image_assets, model_id, dtype: str) -> None:
#     run_vit_vs_intern_vit(
#         image_assets,
#         model_id,
#         dtype=dtype,
#     )
if __name__ == "__main__":
    _test_radio_vllm_vs_hf()

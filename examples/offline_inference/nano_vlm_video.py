#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from dataclasses import asdict
from typing import Optional

from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.assets.image import ImageAsset
from PIL import Image
from vllm.utils import FlexibleArgumentParser
from vision_language import ModelRequestData, apply_image_repeat, get_multi_modal_input


def parse_args():
    parser = FlexibleArgumentParser(
        description="Run nano_vlm image/video inference with vLLM",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "video"],
        default="video",
        help="Input modality.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file. If omitted, uses a sample asset.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to a video file. If omitted, uses a sample asset.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample from the video.",
    )

    parser.add_argument(
        "--num-prompts", type=int, default=4, help="Number of prompts to run."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="/home/dafrimi/projects/models/working_13p41",
        help="Path or HF hub id of the nano_vlm model.",
    )
    parser.add_argument(
        "--seed",
        type=Optional[int],
        default=None,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )
    return parser.parse_args()


def build_prompt_and_stops(model_name: str, question: str, modality: str):
    # Mirror the prompt/stop-token handling from run_nano_vlm in vision_language.py
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    placeholder = "<image>" if modality == "image" else "<video>"
    messages = [[{"role": "user", "content": f"{placeholder}\n{question}"}]]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [t for t in stop_token_ids if t is not None]

    return prompt, stop_token_ids


def main(args):

    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    questions = mm_input["questions"]
    # run_nano_vlm
    engine_args = EngineArgs(
        model=args.model_type,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={args.modality: 1},
        gpu_memory_utilization=0.95,
    )

    prompt, stop_token_ids = build_prompt_and_stops(args.model_type, questions, args.modality)

    req_data = ModelRequestData(
        engine_args=engine_args,
        prompts=prompt,
        stop_token_ids=stop_token_ids,
    )

    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
    }
    llm = LLM(**engine_args)

    # Don't want to check the flag multiple times, so just hijack `prompts`.
    prompts = [req_data.prompts[0]]

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(
        temperature=0.2, max_tokens=64, stop_token_ids=req_data.stop_token_ids)


    if args.num_prompts == 1:
        # Single inference
        inputs = {
            "prompt": prompts[0],
            "multi_modal_data": {args.modality: data},
        }
    else:
        inputs = [
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {args.modality: data},
            }
            for i in range(args.num_prompts)
            ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print("-" * 50)
    for o in outputs:
        print(o.outputs[0].text)
        print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    args.modality = "image"
    args.model_type = "/home/dafrimi/projects/models/working_13p41"
    main(args)



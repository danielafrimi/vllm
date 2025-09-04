# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import PIL

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"
intervl_model = "OpenGVLab/InternVL2-1B"
qwen_vl = "Qwen/Qwen2.5-VL-3B-Instruct"
vlm_ckpt = "/home/dafrimi/projects/models/working_13p41"
llava = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
video_url = (
    "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/"
    "ForBiggerFun.mp4"                    
)

def main():

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
    image = PIL.Image.open("/home/dafrimi/projects/vllm/images/duck.jpg")

    llm = LLM(
        model=qwen_vl,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=5000,
        gpu_memory_utilization=0.95,
    )

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=60)

    outputs = llm.generate(
        {
            "prompt": "what is the content of this video?",
            "multi_modal_data": {
                # "image": image,
                "video": video_url
            }
        },
        sampling_params=sampling_params)

    print("Prompt:", repr(prompt))
    print("Output:", repr(outputs[0].outputs[0].text))


if __name__ == "__main__":
    main()
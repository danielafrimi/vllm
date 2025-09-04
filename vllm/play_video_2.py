from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
video_path = "https://content.pexels.com/videos/free-videos.mp4"

llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    limit_mm_per_prompt={"video": 1},
)

sampling_params = SamplingParams(
    max_tokens=1024,
)

video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "describe this video."},
            {
                "type": "video",
                "video": video_path,
                "total_pixels": 20480 * 28 * 28,
                "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

messages = video_messages
processor = AutoProcessor.from_pretrained(model_path)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)
mm_data = {}
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
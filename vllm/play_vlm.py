from vllm import LLM, SamplingParams
import PIL
import os 
from IPython.display import display
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

os.environ["VLLM_USE_V1"] = "0"

model_hetro = "/home/dafrimi/projects/models/Nvidia-Nemotron-Nano-v2-9B-Flextron-0805"
model_homo = "/home/dafrimi/projects/models/Nvidia-Nemotron-Nano-v2-9B-0805"
model_nemtoron = "/opt/data/llm-models/Nemotron-H-8B-Base-8K"
# b  = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
tzag_vlm = "/home/dafrimi/projects/models/working_13p41"


def main():

    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"
    image = PIL.Image.open("/home/dafrimi/projects/vllm/images/duck.jpg")

    llm = LLM(
        model=tzag_vlm,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,  max_model_len=4000,
        gpu_memory_utilization=0.95,
         )
        
    


    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=60)

    # # prompt = "The future of AI is"
    # outputs = llm.generate(prompt, sampling_params)

    outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image} }, sampling_params=sampling_params
)


    print("Prompt:", repr(prompt))
    print("Output:", repr(outputs[0].outputs[0].text))


if __name__ == "__main__":
    main()
from vllm import LLM, SamplingParams


vlm_model = "/lustre/fsw/portfolios/adlr/projects/adlr_nlp_llmnext/ksapra/weights/nvlm_v2_1341/mcore_to_hf_fixed"
vlm_model_peter = "/lustre/fsw/portfolios/llmservice/users/charlwang/nvwork/250709_vlm/model_ckpt/hf_fixed_sft_v1338_32k_iter_22000"


def main():
    llm = LLM(
        model=vlm_model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        )
        
    

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=60)

    prompt = "The future of AI is"
    outputs = llm.generate(prompt, sampling_params)

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
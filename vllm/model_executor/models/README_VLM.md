
2.   create venv 
cd <path to your workspace>
python -m venv .vllm
. .vllm/bin/activate
pip install -U pip setuptools packaging wheel ninja 'cmake<4' 

1. git checkout branch and clone
2. cd vllm 
in vevn: 
3. VLLM_USE_PRECOMPILED=1 pip install -e .
4. pip install -U 'transformers<4.54' timm open_clip_torch 
5. pip install mamba-ssm no isolatoin ..............


We assume you already have a HF version of your Nano V2 VLM checkpoint.
Once you have this HF checkpoint, you will need to prepare the checkpoint metadata:

Fixup the HF checkpoint config.json


For compatibility with the VLM inference code in this repository
(i.e. custom_vlm.py), please set the checkpoint config "model_type" to
the value "NemotronH_Nano_VL".


If your config.json has a top-level dictionary key named "llm_config",
please rename it to "text_config".


Inside the dictionary-typed value corresponding to key "text_config",
there should be a subkey "hybrid_override_pattern".
This is a shorthand pattern for specifying the layer types in the hybrid
model architecture (i.e. attention, mamba, or mlp).
We need to convert this to a full layer block type spec, for which
we have provided a helper script in the file convert_layers_block_type.py
(located in llama_nemotron_nano_vl_vllm-dev/n5h_nano_vl/vision_lm,
i.e. in the same directory as this README).
To run the helper script:

python3 convert_layers_block_type.py --config-path=<path to HF checkpoint config.json> --dump-full-config


This above command will dump (to stdout) a patched version of the original
config.json;
the patched config which will be compatible w/ our VLLM branch.
Then replace the original config.json with the patched config dumped by the
above command.




maybe:


change configuration attrs in config.json
model_type --> NemotronH_Nano_VL
in modeling.py:
all places written with   NemotronH_Nano_VL_V2_Config change with NemotronH_Nano_VL_Config
in coniguraation.py change:
change class name is NemotronH_Nano_VL_Config
model type is model_type = 'NemotronH_Nano_VL'
refactor  llm_config  with text_config (change the names)
init get a text_config (remove the llm_config) and add the following: 
        if text_config is None:
            text_config = {}
        self.text_config = PretrainedConfig(**text_config)


example script can be found play_vlm.py


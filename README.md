# Decomposed LLama

There is a LLama model code adopted for better debuging in Jupyter Notebook.

Llama code is taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py and splitted into several files by classes. And after that an adoption is applied to LlamaAttention class to demonstration the technique.

# Register Llama model

To use Llama model class from this repo instead fo transofrmers' one register it.

First of all download Llama model, for example https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="config.json", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="generation_config.json", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="model-00001-of-00002.safetensors", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="model-00002-of-00002.safetensors", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="model.safetensors.index.json", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="special_tokens_map.json", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="tokenizer.json", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="tokenizer.model", local_dir="./", local_dir_use_symlinks=False)
hf_hub_download(repo_id="meta-llama/Llama-2-7b-chat-hf", filename="tokenizer_config.json", local_dir="./", local_dir_use_symlinks=False)
```

Adjust config.json from the model directory by changing "model_type" field from "llama" to "llama_decomposed".
```json
    ...
    "model_type": "llama_decomposed",
    ...
```

Register model with next code
```
from llm.model.LlamaForCasualLM import LlamaForCasualLM

LlamaForCasualLM.register_model()
```

# Load model with repo class

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # other args
)
```

`model_path` here should point to the model directory with adjusted config.json file

Use better code for debug and development.
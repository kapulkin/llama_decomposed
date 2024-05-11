from huggingface_hub import hf_hub_download

model_id = "meta-llama/Llama-2-7b-chat-hf"
model_dir = "models/"

hf_hub_download(repo_id=model_id, filename="config.json", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="generation_config.json", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="model-00001-of-00002.safetensors", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="model-00002-of-00002.safetensors", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="special_tokens_map.json", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="tokenizer.json", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="tokenizer.model", local_dir=model_dir, local_dir_use_symlinks=False)
hf_hub_download(repo_id=model_id, filename="tokenizer_config.json", local_dir=model_dir, local_dir_use_symlinks=False)

from transformers.models.llama.configuration_llama import LlamaConfig

class LlamaModelConfig(LlamaConfig):
    model_type = "llama_decomposed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

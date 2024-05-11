from llm.model.LlamaForCasualLM import LlamaForCasualLM

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, TrainingArguments, Trainer, \
    AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig, \
    PreTrainedTokenizer
from functools import partial
import torch
import copy

IGNORE_INDEX = -100


def batch_data_collator(pad_values, samples):
    batch = {}
    first = samples[0]
    for key in first:
        batch[key] = pad_sequence([
            torch.tensor(s[key])
            for s in samples
        ], batch_first=True, padding_value=pad_values[key])
    return batch

def main():
    LlamaForCasualLM.register_model()

    model_path = "models/llama"

    model_args = {
        "quantization_config": BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        ),
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True
    }
    model: LlamaForCasualLM = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
    
    model.enable_parallel_pipeline(2)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    batch_size = 2

    dataset = Dataset.from_list([
        { "input": "test1", "label": "test1" },
        { "input": "test2", "label": "test2" },
        { "input": "test3", "label": "test3" },
        { "input": "test4", "label": "test4" },
    ])

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        remove_columns=["input", "label"],
        desc='Tokenize',
        fn_kwargs={
            'tokenizer': tokenizer
        },
    )

    data_collator = partial(batch_data_collator, {
        'input_ids': tokenizer.pad_token_id,
        'attention_mask': False,
        'labels': IGNORE_INDEX
    })

    training_args = TrainingArguments(
        output_dir="output",
        save_strategy='steps',
        save_steps=25,
        push_to_hub=False,
        per_device_train_batch_size=batch_size,
        bf16=False,
        fp16=False,
        learning_rate=0.00002,
        optim="paged_adamw_8bit",
    )
    
    trainer = Trainer(
        args=training_args,
        train_dataset=dataset,
        model=model,
        data_collator=data_collator,
    )

    trainer.train()

def tokenize(batch, tokenizer: PreTrainedTokenizer):
    result_batch = {}

    batch_size = len(next(iter(batch.values())))
    for i in range(batch_size):
        entry = { key: values[i] for key, values in batch.items()}
        
        prompt = entry["input"]
        label = entry["label"]

        source = f"{tokenizer.bos_token}{prompt}"
        target = f"{label}{tokenizer.eos_token}"
        # Tokenize
        tokenized_source_with_prompt = tokenizer(
            source,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_target = tokenizer(
            target,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        tokenized_source_ids = tokenized_source_with_prompt['input_ids']
        tokenized_target_ids = tokenized_target['input_ids']
        input_ids = torch.tensor(tokenized_source_ids + tokenized_target_ids)
        labels = torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source_ids))] + copy.deepcopy(tokenized_target_ids))
        
        
        data = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(tokenizer.pad_token_id),
            'labels': labels
        }
        
        for key, value in data.items():
            if key not in result_batch:
                result_batch[key] = []
            result_batch[key].append(value)

    return result_batch
        
if __name__ == "__main__":
    main()
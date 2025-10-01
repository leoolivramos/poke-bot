import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "../data/processed/"
output_dir = "../models/tiny-lama-lora-pokemon"

dataset = load_dataset("json", data_dir=dataset_path, split="train")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    overwrite_output_dir=True,
)

def formatting_prompts_func(example):
    text = f"<s>[INST] {example['instruction']} [/INST] {example['output']} </s>"
    return text

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
)

print("Iniciando o fine-tuning...")
trainer.train()

print("Salvando o modelo treinado...")
trainer.save_model(output_dir)

print("Treinamento concluído com sucesso!")
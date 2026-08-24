from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Resolver caminhos absolutos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "tiny-lama-lora-pokemon"

print(f"Carregando dataset de: {DATASET_PATH}")
dataset = load_dataset("json", data_dir=str(DATASET_PATH), split="train")

# Configurar quantização apenas se houver GPU CUDA disponível
has_cuda = torch.cuda.is_available()
bnb_config = None

if has_cuda:
    print("GPU CUDA detectada. Aplicando quantização 4-bit (BitsAndBytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
else:
    print("CUDA não disponível. O treinamento rodará em modo padrão (CPU/Float32).")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" if has_cuda else None,
    trust_remote_code=True,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
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

# A LoraConfig é passada diretamente para o SFTTrainer, que cuidará da preparação do PeftModel


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=has_cuda,
    dataloader_pin_memory=has_cuda,
    save_total_limit=2
)

def formatting_prompts_func(example):
    """Formato oficial do TinyLlama-Chat: <|user|> ... <|assistant|>"""
    return f"<s><|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['output']}</s>"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
)

print("Iniciando o fine-tuning...")
trainer.train()

print(f"Salvando o modelo treinado em {OUTPUT_DIR}...")
trainer.save_model(str(OUTPUT_DIR))

print("Treinamento concluído com sucesso!")

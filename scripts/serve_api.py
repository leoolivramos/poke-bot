from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "tiny-lama-lora-pokemon"))

app = FastAPI(
    title="Pokémon LLM API",
    description="API para responder perguntas sobre Pokémon usando um modelo fine-tunado."
)

print("Iniciando o carregamento do modelo...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Carregando modelo base: {base_model_id}...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Configurar token de padding

# 4. Carregar e aplicar o adaptador LoRA treinado sobre o modelo base
print(f"Carregando e aplicando o adaptador LoRA de: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("✅ Modelo carregado com sucesso e pronto para receber requisições!")

# --- Definição do Endpoint ---
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_pokemon(query: Query):
    """Recebe uma pergunta sobre Pokémon e retorna a resposta gerada pelo modelo."""
    
    # Formata o prompt para o estilo do Mistral Instruct
    prompt = f"<s>[INST] {query.question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Gera a resposta
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7,
            do_sample=True,
        )
        
    # Decodifica a resposta e limpa o prompt inicial
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response_text.split("[/INST]")[-1].strip()

    return {"question": query.question, "answer": answer}
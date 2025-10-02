from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
from contextlib import asynccontextmanager

# Dicionário para armazenar os modelos carregados durante o lifespan
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que roda ANTES de a API começar a receber requisições
    print("Iniciando o carregamento do modelo...")
    
    # --- Configuração dos Modelos ---
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "tiny-lama-lora-pokemon"))

    # 1. Carregar o modelo base (TinyLlama)
    print(f"Carregando modelo base: {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )

    # 2. Carregar o Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Carregar e aplicar o adaptador LoRA
    print(f"Carregando e aplicando o adaptador LoRA de: {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Armazena o modelo e o tokenizer no dicionário
    ml_models["model"] = model
    ml_models["tokenizer"] = tokenizer
    
    print("✅ Modelo carregado com sucesso!")
    
    yield
    
    # Código que roda DEPOIS que a API é desligada (limpeza)
    ml_models.clear()
    print("Modelos descarregados.")

# Cria a aplicação FastAPI e associa o lifespan
app = FastAPI(
    lifespan=lifespan,
    title="Pokémon LLM API",
    description="API para responder perguntas sobre Pokémon usando um modelo fine-tunado."
)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_pokemon(query: Query):
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    
    prompt = f"<s><|user|>\n{query.question}</s>\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7,
            do_sample=True,
        )
        
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response_text.split("<|assistant|>")[-1].strip()

    return {"question": query.question, "answer": answer}
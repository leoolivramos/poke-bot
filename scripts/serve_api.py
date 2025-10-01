from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "mistral-lora-pokemon"))

app = FastAPI(
    title="Pokémon LLM API",
    description="API para responder perguntas sobre Pokémon usando um modelo fine-tunado."
)

print("Iniciando o carregamento do modelo...")

print(f"Carregando modelo base: {base_model_id}...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"Carregando e aplicando o adaptador LoRA de: {adapter_path}...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("✅ Modelo carregado com sucesso e pronto para receber requisições!")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_pokemon(query: Query):
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
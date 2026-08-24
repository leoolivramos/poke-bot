from contextlib import asynccontextmanager
import json
from pathlib import Path
import os
import sys
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Estrutura global para guardar modelos e estado da API
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = PROJECT_ROOT / "models" / "tiny-lama-lora-pokemon"
POKEMON_DB_PATH = PROJECT_ROOT / "data" / "processed" / "pokemon_db.json"

ml_models = {}
pokemon_db = {}

def load_pokemon_db():
    """Carrega o banco estruturado de Pokémon se existir."""
    global pokemon_db
    if POKEMON_DB_PATH.exists():
        try:
            with open(POKEMON_DB_PATH, "r", encoding="utf-8") as f:
                pokemon_db = json.load(f)
            print(f"📊 Banco estruturado de Pokémon carregado ({len(pokemon_db)} registros).")
        except Exception as e:
            print(f"⚠️ Erro ao carregar pokemon_db.json: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando o carregamento dos recursos da API...")
    load_pokemon_db()

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    adapter_loaded = False

    try:
        print(f"📦 Carregando modelo base: {BASE_MODEL_ID} em [{device}]...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if has_cuda else torch.float32,
            device_map="auto" if has_cuda else None,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Tentar carregar adaptador LoRA se existir
        adapter_config = ADAPTER_PATH / "adapter_config.json"
        if adapter_config.exists():
            print(f"🎯 Aplicando adaptador LoRA de: {ADAPTER_PATH}...")
            model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
            adapter_loaded = True
        else:
            print(f"⚠️ Adaptador LoRA não encontrado em {ADAPTER_PATH}. Usando modelo base TinyLlama.")
            model = base_model

        if not has_cuda:
            model = model.to("cpu")

        ml_models["model"] = model
        ml_models["tokenizer"] = tokenizer
        ml_models["device"] = device
        ml_models["adapter_loaded"] = adapter_loaded
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo LLM: {e}")
        ml_models["model"] = None
        ml_models["tokenizer"] = None
        ml_models["device"] = device
        ml_models["adapter_loaded"] = False

    yield

    ml_models.clear()
    print("🧹 Recusos da API descarregados.")

app = FastAPI(
    lifespan=lifespan,
    title="Pokédex IA API",
    description="API de IA Conversacional e Consulta de Dados Pokémon",
    version="2.0.0"
)

class AskQuery(BaseModel):
    question: str = Field(..., example="Fale sobre o Pikachu")
    temperature: float = Field(default=0.7, ge=0.1, le=1.5)
    max_tokens: int = Field(default=150, ge=10, le=500)

@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à Pokédex IA API! Acesse /docs para ver os endpoints disponíveis.",
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": ml_models.get("device", "unknown"),
        "model_loaded": ml_models.get("model") is not None,
        "adapter_loaded": ml_models.get("adapter_loaded", False),
        "total_pokemon_db": len(pokemon_db) // 2 if pokemon_db else 0
    }

@app.get("/pokemon/{query}")
async def get_pokemon_info(query: str):
    """Busca ficha técnica de um Pokémon por nome ou ID."""
    key = query.strip().lower()
    if key in pokemon_db:
        return pokemon_db[key]
    raise HTTPException(status_code=404, detail=f"Pokémon '{query}' não encontrado no banco local.")

@app.post("/ask")
async def ask_pokemon(query: AskQuery):
    """Responde a perguntas sobre Pokémon usando a IA conversacional."""
    model = ml_models.get("model")
    tokenizer = ml_models.get("tokenizer")

    # Fallback se o modelo de LLM não estiver carregado (ex: falta de memória)
    if model is None or tokenizer is None:
        # Tentar buscar resposta rápida no banco local se mencionar algum Pokémon
        q_lower = query.question.lower()
        matched = [p for name, p in pokemon_db.items() if isinstance(p, dict) and p.get("name", "").lower() in q_lower]
        if matched:
            p = matched[0]
            answer = f"({p['name']} - #{p['id']}) {p['description']} Tipos: {', '.join(p['types'])}."
            return {"question": query.question, "answer": answer, "mode": "fallback_db"}
        return {"question": query.question, "answer": "O modelo de IA está indisponível no momento.", "mode": "error"}

    prompt = f"<s><|user|>\n{query.question}</s>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(ml_models["device"])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=query.max_tokens,
            temperature=query.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response_text.split("<|assistant|>")[-1].strip()

    return {"question": query.question, "answer": answer, "mode": "llm"}
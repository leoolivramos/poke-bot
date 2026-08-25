import asyncio
import json
import random
import sys
import time
from pathlib import Path
from deep_translator import MyMemoryTranslator
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://pokeapi.co/api/v2/pokemon/"
SPECIES_URL = "https://pokeapi.co/api/v2/pokemon-species/"
TOTAL_POKEMON = 155
CONCURRENCY_LIMIT = 20

# Garantir caminho absoluto da pasta data/processed na raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
translator = MyMemoryTranslator(source="en-US", target="pt-BR")
translation_cache = {}
TYPE_TRANSLATIONS = {
    "bug": "Inseto",
    "dark": "Sombrio",
    "dragon": "Dragão",
    "electric": "Elétrico",
    "fairy": "Fada",
    "fighting": "Lutador",
    "fire": "Fogo",
    "flying": "Voador",
    "ghost": "Fantasma",
    "grass": "Grama",
    "ground": "Terrestre",
    "ice": "Gelo",
    "normal": "Normal",
    "poison": "Veneno",
    "psychic": "Psíquico",
    "rock": "Pedra",
    "steel": "Aço",
    "water": "Água",
}

def translate_text(text: str) -> str:
    """Traduz um texto da PokéAPI para português brasileiro."""
    if not text or text in translation_cache:
        return translation_cache.get(text, text)

    for attempt in range(3):
        try:
            time.sleep(0.3)
            translated = translator.translate(text)
            if translated and not translated.lower().startswith(("error ", "server error")):
                translation_cache[text] = translated
                return translated
        except Exception:
            if attempt == 2:
                break
            time.sleep(1)

    return text

def translate_type(type_name: str) -> str:
    """Traduz os tipos conhecidos sem depender de um serviço externo."""
    return TYPE_TRANSLATIONS.get(type_name, translate_text(type_name))

async def fetch_pokemon(client: httpx.AsyncClient, sem: asyncio.Semaphore, pokemon_id: int):
    """Busca dados brutos do Pokémon e da espécie simultaneamente na PokéAPI."""
    async with sem:
        try:
            p_resp = await client.get(f"{BASE_URL}{pokemon_id}", timeout=15.0)
            if p_resp.status_code != 200:
                return None
            p_data = p_resp.json()

            s_resp = await client.get(f"{SPECIES_URL}{pokemon_id}", timeout=15.0)
            s_data = s_resp.json() if s_resp.status_code == 200 else {}

            return parse_pokemon(p_data, s_data)
        except Exception as err:
            print(f"Erro ao buscar Pokémon #{pokemon_id}: {err}")
            return None

def parse_pokemon(data: dict, species_data: dict) -> dict:
    """Extrai informações estruturadas do Pokémon."""
    p_id = data.get("id")
    name = data.get("name", "").capitalize()
    
    types = [translate_type(type_data["type"]["name"]) for type_data in data.get("types", [])]
    abilities = [translate_text(a["ability"]["name"].replace("-", " ")) for a in data.get("abilities", [])]
    
    # Imagem oficial
    sprites = data.get("sprites", {})
    official_art = sprites.get("other", {}).get("official-artwork", {}).get("front_default")
    image_url = official_art or sprites.get("front_default") or ""

    # Stats
    stats = {}
    for s in data.get("stats", []):
        stat_name = s["stat"]["name"]
        stats[stat_name] = s["base_stat"]

    height = data.get("height", 0) / 10.0  # decímetros para metros
    weight = data.get("weight", 0) / 10.0  # hectogramas para kg

    # Descrição (Flavor Text)
    description = "Nenhuma descrição encontrada."
    for entry in species_data.get("flavor_text_entries", []):
        lang = entry.get("language", {}).get("name")
        if lang in ["pt", "en"]:
            description = entry["flavor_text"].replace("\n", " ").replace("\f", " ").strip()
            description = translate_text(description) if lang == "en" else description
            if lang == "pt":
                break  # Prefere português se disponível

    return {
        "id": p_id,
        "name": name,
        "types": types,
        "abilities": abilities,
        "stats": stats,
        "height": height,
        "weight": weight,
        "image_url": image_url,
        "description": description,
    }

def generate_qa_pairs(info: dict) -> list:
    """Gera pares de pergunta e resposta com variações ricas."""
    if not info:
        return []

    name = info["name"]
    types_str = " e ".join(info["types"])
    abilities_str = ", ".join(info["abilities"])
    desc = info["description"]
    
    hp = info["stats"].get("hp", "N/A")
    attack = info["stats"].get("attack", "N/A")
    defense = info["stats"].get("defense", "N/A")
    speed = info["stats"].get("speed", "N/A")

    pairs = [
        {
            "instruction": f"Qual é o tipo do {name}?",
            "output": f"{name} é um Pokémon do tipo {types_str}."
        },
        {
            "instruction": f"Quais são os tipos do {name}?",
            "output": f"Os tipos do {name} são {types_str}."
        },
        {
            "instruction": f"Fale sobre o {name}.",
            "output": f"{name} é um Pokémon do tipo {types_str}. {desc}"
        },
        {
            "instruction": f"Descreva o {name}.",
            "output": f"{desc} Ele é do tipo {types_str}, mede {info['height']}m e pesa {info['weight']}kg."
        },
        {
            "instruction": f"Quais são as habilidades do {name}?",
            "output": f"As habilidades do {name} são: {abilities_str}."
        },
        {
            "instruction": f"Quais os atributos e stats do {name}?",
            "output": f"{name} possui HP: {hp}, Ataque: {attack}, Defesa: {defense} e Velocidade: {speed}."
        }
    ]
    return pairs

async def create_dataset():
    """Baixa os dados de forma assíncrona, gera o dataset e salva os arquivos JSONL e JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"Iniciando download assíncrono de {TOTAL_POKEMON} Pokémon da PokéAPI...")
    async with httpx.AsyncClient() as client:
        tasks = [fetch_pokemon(client, sem, i) for i in range(1, TOTAL_POKEMON + 1)]
        results = await asyncio.gather(*tasks)

    valid_pokemon = [p for p in results if p is not None]
    print(f"{len(valid_pokemon)} Pokémon coletados com sucesso!")

    # Salvar base de dados JSON estruturada para uso rápido pela API/Bot
    db_file = OUTPUT_DIR / "pokemon_db.json"
    db_data = {p["name"].lower(): p for p in valid_pokemon}
    db_data.update({str(p["id"]): p for p in valid_pokemon})
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(db_data, f, ensure_ascii=False, indent=2)

    # Gerar QA pairs
    all_qa_pairs = []
    for p in valid_pokemon:
        all_qa_pairs.extend(generate_qa_pairs(p))

    random.shuffle(all_qa_pairs)
    split_index = int(len(all_qa_pairs) * 0.8)
    train_data = all_qa_pairs[:split_index]
    valid_data = all_qa_pairs[split_index:]

    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset criado com sucesso!")
    print(f" - Treino: {len(train_data)} exemplos em {OUTPUT_DIR / 'train.jsonl'}")
    print(f" - Validação: {len(valid_data)} exemplos em {OUTPUT_DIR / 'valid.jsonl'}")
    print(f" - Banco de dados estruturado salvo em {db_file}")

if __name__ == "__main__":
    asyncio.run(create_dataset())

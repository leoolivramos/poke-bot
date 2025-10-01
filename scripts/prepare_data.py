import requests
import json
import random
import os

BASE_URL = "https://pokeapi.co/api/v2/pokemon/"
SPECIES_URL = "https://pokeapi.co/api/v2/pokemon-species/"
TOTAL_POKEMON = 151

def get_pokemon_data(pokemon_id):
    """Busca dados de um Pokémon específico na PokéAPI."""
    try:
        response = requests.get(f"{BASE_URL}{pokemon_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"Erro ao buscar dados do Pokémon {pokemon_id}: {err}")
        return None

def get_species_description(pokemon_id):
    """Busca a descrição da Pokédex (flavor text)."""
    try:
        response = requests.get(f"{SPECIES_URL}{pokemon_id}")
        response.raise_for_status()
        species_data = response.json()
        for entry in species_data.get('flavor_text_entries', []):
            if entry['language']['name'] == 'pt' or entry['language']['name'] == 'en':
                return entry['flavor_text'].replace('\n', ' ').replace('\f', ' ')
        return "Nenhuma descrição encontrada."
    except requests.exceptions.HTTPError:
        return "Nenhuma descrição encontrada."


def generate_qa_pairs(pokemon_data):
    """Gera pares de pergunta e resposta a partir dos dados do Pokémon."""
    if not pokemon_data:
        return []

    name = pokemon_data['name'].capitalize()
    types = [t['type']['name'].capitalize() for t in pokemon_data['types']]
    abilities = [a['ability']['name'].capitalize() for a in pokemon_data['abilities']]
    description = get_species_description(pokemon_data['id'])

    if len(types) > 1:
        types_str = f"{' e '.join(types)}"
    else:
        types_str = types[0]
    
    abilities_str = ', '.join(abilities)

    pairs = [
        {
            "instruction": f"Qual é o tipo do {name}?",
            "output": f"{name} é do tipo {types_str}."
        },
        {
            "instruction": f"Quais são os tipos do {name}?",
            "output": f"Os tipos do {name} são {types_str}."
        },
        {
            "instruction": f"Fale sobre o {name}.",
            "output": f"{name} é um Pokémon do tipo {types_str}. {description}"
        },
        {
            "instruction": f"Descreva o {name}.",
            "output": f"{description} Seus tipos são {types_str} e suas habilidades incluem {abilities_str}."
        },
        {
            "instruction": f"Quais as habilidades do {name}?",
            "output": f"As habilidades do {name} são: {abilities_str}."
        }
    ]
    return pairs


def create_dataset():
    """Cria o dataset completo e salva em train.jsonl e valid.jsonl."""
    output_dir = "../data/processed"

    os.makedirs(output_dir, exist_ok=True)

    all_qa_pairs = []
    for i in range(1, TOTAL_POKEMON + 1):
        print(f"Processando Pokémon #{i}...")
        data = get_pokemon_data(i)
        if data:
            all_qa_pairs.extend(generate_qa_pairs(data))
    
    random.shuffle(all_qa_pairs)
    
    split_index = int(len(all_qa_pairs) * 0.8)
    train_data = all_qa_pairs[:split_index]
    valid_data = all_qa_pairs[split_index:]

    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(os.path.join(output_dir, "valid.jsonl"), "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Dataset criado com sucesso! {len(train_data)} exemplos de treino e {len(valid_data)} de validação.")

if __name__ == "__main__":
    create_dataset()
import pytest
from fastapi.testclient import TestClient
from scripts.serve_api import app, pokemon_db
from scripts.prepare_data import parse_pokemon, generate_qa_pairs

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data

def test_parse_pokemon_and_qa():
    mock_pokemon_raw = {
        "id": 25,
        "name": "pikachu",
        "types": [{"type": {"name": "electric"}}],
        "abilities": [{"ability": {"name": "static"}}],
        "stats": [{"stat": {"name": "hp"}, "base_stat": 35}],
        "height": 4,
        "weight": 60,
        "sprites": {"front_default": "http://example.com/pikachu.png"}
    }
    mock_species_raw = {
        "flavor_text_entries": [
            {"flavor_text": "When several of these POKéMON gather, their electricity could build...", "language": {"name": "en"}}
        ]
    }

    parsed = parse_pokemon(mock_pokemon_raw, mock_species_raw)
    assert parsed["name"] == "Pikachu"
    assert parsed["types"] == ["Electric"]
    assert parsed["height"] == 0.4
    assert parsed["weight"] == 6.0

    qa_pairs = generate_qa_pairs(parsed)
    assert len(qa_pairs) > 0
    assert any("Pikachu" in pair["output"] for pair in qa_pairs)

def test_pokemon_lookup_endpoint():
    # Popula mock temporário no pokemon_db
    pokemon_db["pikachu"] = {
        "id": 25,
        "name": "Pikachu",
        "types": ["Electric"],
        "abilities": ["Static"],
        "stats": {"hp": 35, "attack": 55},
        "height": 0.4,
        "weight": 6.0,
        "image_url": "http://example.com/pikachu.png",
        "description": "Mouse Pokémon"
    }

    response = client.get("/pokemon/pikachu")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Pikachu"
    assert data["id"] == 25

def test_pokemon_not_found():
    response = client.get("/pokemon/nonexistent_pokemon")
    assert response.status_code == 404

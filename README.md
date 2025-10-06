# Pokédex IA (Telegram)

Projeto: Pokédex conversacional + API de inferência.

Conteúdo:
- `prepare_data.py`: coleta da PokéAPI e geração de dataset JSONL (instrução -> resposta).
- `train_lora.py`: esqueleto para fine-tuning com PEFT + BitsAndBytes (LoRA + 4-bit quant).
- `serve_api.py`: FastAPI simples que expõe `/ask` (integra com seu modelo).
- `telegram_bot.py`: bot Telegram que consulta a API e responde.

## Quickstart

1. Crie e ative virtualenv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Copie .env.example para .env e ajuste
```bash
TELEGRAM_TOKEN=...
API_URL=http://localhost:8000/ask
```
3. Rodar API
```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```
4. Rodar bot
```bash
python telegram_bot.py
```
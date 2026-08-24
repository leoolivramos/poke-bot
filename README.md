# Pokédex IA (Telegram & API)

Aplicação completa de Pokédex conversacional com IA, API FastAPI e Bot do Telegram interativo.

## 🚀 Recursos

- **🤖 Bot Telegram Interativo**:
  - Cards com artes oficiais dos Pokémon.
  - Teclado com botões interativos (Inline Keyboards).
  - Comandos: `/start`, `/help`, `/pokemon <nome_ou_id>`, `/random`.
  - Respostas assíncronas via `httpx` sem bloquear a interface.
- **⚡ Ingestão Rápida de Dados (`prepare_data.py`)**:
  - Download assíncrono paralelo via `httpx` + `asyncio`.
  - Extração de artes oficiais, atributos base (HP, Ataque, Defesa, etc.), peso, altura e descrições em PT/EN.
- **🧠 Fine-Tuning LoRA (`train_lora.py`)**:
  - Fine-tuning com PEFT (LoRA) + BitsAndBytes (4-bit quantization) sobre o modelo **TinyLlama-1.1B-Chat**.
  - Template de prompt padronizado no formato nativo do TinyLlama (`<|user|>` e `<|assistant|>`).
- **🛡️ API FastAPI Resiliente (`serve_api.py`)**:
  - Fallback gracioso para modelo base ou banco local se o adaptador LoRA não estiver presente.
  - Endpoints: `POST /ask`, `GET /pokemon/{query}`, `GET /health`.
- **🐳 Docker & Docker Compose**:
  - Orquestração pronta para subir a API e o Bot em contêineres isolados.
- **🧪 Testes Automatizados (`pytest`)**:
  - Cobertura de testes unitários para a API e o pipeline de dados.

---

## 🛠️ Quickstart

### 1. Instalação Local

```bash
# Criar e ativar virtualenv
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Variáveis de Ambiente

Copie `.env.example` para `.env` e defina seu token do Telegram:
```env
TELEGRAM_TOKEN=seu_token_aqui
API_URL=http://localhost:8000
```

### 3. Gerar o Dataset da PokéAPI

```bash
python scripts/prepare_data.py
```

### 4. Executar a API e o Bot

**Terminal 1 (API):**
```bash
uvicorn scripts.serve_api:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Bot Telegram):**
```bash
python bot/telegram_bot.py
```

---

## 🐳 Execução via Docker Compose

Suba a API e o Bot do Telegram com um único comando:
```bash
docker-compose up --build
```

---

## 🧪 Testes

Para rodar a suíte de testes automatizados:
```bash
pytest tests/
```
import os
import random
from pathlib import Path
import httpx
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Carregar variáveis de ambiente de .env se existir
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "SEU_TELEGRAM_TOKEN_AQUI")
API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_main_keyboard():
    """Retorna teclado interativo padrão."""
    keyboard = [
        [
            InlineKeyboardButton("Pikachu", callback_data="poke_pikachu"),
            InlineKeyboardButton("Charizard", callback_data="poke_charizard"),
            InlineKeyboardButton("Blastoise", callback_data="poke_blastoise"),
        ],
        [
            InlineKeyboardButton("Pokémon Aleatório", callback_data="poke_random"),
            InlineKeyboardButton("Ajuda", callback_data="help_menu"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /start"""
    welcome_msg = (
        "**Olá! Eu sou a Pokédex IA!**\n\n"
        "Você pode:\n"
        "• Me fazer qualquer pergunta no chat (ex: *'Qual o tipo do Bulbasaur?'*)\n"
        "• Usar o comando `/pokemon <nome_ou_id>` para ver a ficha completa com foto!\n"
        "• Usar `/random` para sortear um Pokémon!\n"
        "• Ou clicar em um dos botões abaixo:"
    )
    await update.message.reply_text(
        welcome_msg, parse_mode="Markdown", reply_markup=get_main_keyboard()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /help"""
    help_text = (
        "**Guia da Pokédex IA**\n\n"
        "🔹 **Perguntas livres**: Escreva qualquer dúvida no chat.\n"
        "🔹 `/pokemon <nome_ou_id>`: Mostra imagem oficial, status, tipos e peso/altura.\n"
        "🔹 `/random`: Escolhe um Pokémon aleatório entre os 151 clássicos.\n"
        "🔹 `/help`: Exibe este menu de ajuda.\n"
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode="Markdown")
    elif update.callback_query:
        await update.callback_query.message.reply_text(help_text, parse_mode="Markdown")

async def send_pokemon_card(update_or_query, pokemon_data: dict):
    """Formata e envia o card visual do Pokémon com foto e atributos."""
    name = pokemon_data.get("name", "Desconhecido")
    p_id = pokemon_data.get("id", "?")
    types = ", ".join(pokemon_data.get("types", []))
    abilities = ", ".join(pokemon_data.get("abilities", []))
    height = pokemon_data.get("height", 0)
    weight = pokemon_data.get("weight", 0)
    desc = pokemon_data.get("description", "")
    image_url = pokemon_data.get("image_url")
    stats = pokemon_data.get("stats", {})

    stats_text = (
        f"❤️ HP: {stats.get('hp', 'N/A')} | ⚔️ Atq: {stats.get('attack', 'N/A')} | 🛡️ Def: {stats.get('defense', 'N/A')}\n"
        f"⚡ Vel: {stats.get('speed', 'N/A')} | ✨ Sp.Atq: {stats.get('special-attack', 'N/A')}"
    )

    caption = (
        f"🔴 **#{p_id} {name}**\n\n"
        f"🏷️ **Tipos**: {types}\n"
        f"🌟 **Habilidades**: {abilities}\n"
        f"📏 **Altura**: {height}m | ⚖️ **Peso**: {weight}kg\n\n"
        f"📊 **Atributos Base**:\n{stats_text}\n\n"
        f"📝 **Descrição**: {desc}"
    )

    target_message = update_or_query.message if hasattr(update_or_query, "message") and update_or_query.message else update_or_query.callback_query.message

    if image_url:
        try:
            await target_message.reply_photo(
                photo=image_url, caption=caption, parse_mode="Markdown"
            )
            return
        except Exception:
            pass  # Fallback para mensagem de texto pura se der erro no envio de imagem

    await target_message.reply_text(caption, parse_mode="Markdown")

async def pokemon_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /pokemon <nome_ou_id>"""
    if not context.args:
        await update.message.reply_text("⚠️ Por favor informe o nome ou ID do Pokémon. Exemplo: `/pokemon charizard`", parse_mode="Markdown")
        return

    query = context.args[0].lower().strip()
    await update.message.chat.send_action(action="typing")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{API_URL}/pokemon/{query}", timeout=10.0)
            if resp.status_code == 200:
                await send_pokemon_card(update, resp.json())
            else:
                await update.message.reply_text(f"❌ Pokémon '{query}' não encontrado!")
        except Exception as e:
            await update.message.reply_text(f"⚠️ Erro ao comunicar com a API: {e}")

async def random_pokemon(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando /random ou callback botão aleatório."""
    random_id = random.randint(1, 151)
    target = update.message if update.message else update.callback_query.message
    await target.chat.send_action(action="typing")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{API_URL}/pokemon/{random_id}", timeout=10.0)
            if resp.status_code == 200:
                await send_pokemon_card(update, resp.json())
            else:
                await target.reply_text(f"❌ Erro ao buscar Pokémon #{random_id}.")
        except Exception as e:
            await target.reply_text(f"⚠️ Erro ao comunicar com a API: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trata cliques nos botões do teclado interativo."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if data == "help_menu":
        await help_command(update, context)
    elif data == "poke_random":
        await random_pokemon(update, context)
    elif data.startswith("poke_"):
        poke_name = data.split("poke_")[1]
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{API_URL}/pokemon/{poke_name}", timeout=10.0)
                if resp.status_code == 200:
                    await send_pokemon_card(update, resp.json())
                else:
                    await query.message.reply_text(f"❌ Pokémon {poke_name} não encontrado.")
            except Exception as e:
                await query.message.reply_text(f"⚠️ Erro: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trata mensagens de texto do usuário usando a API conversacional /ask."""
    user_message = update.message.text
    await update.message.chat.send_action(action="typing")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{API_URL}/ask",
                json={"question": user_message},
                timeout=60.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "Desculpe, não consegui obter uma resposta.")
            else:
                answer = f"⚠️ Erro na API ({resp.status_code})."
        except Exception as e:
            answer = f"❌ Não foi possível conectar à API de Pokédex ({e}). Verifique se a API está rodando!"

    await update.message.reply_text(answer, reply_markup=get_main_keyboard())

def main():
    if TELEGRAM_TOKEN == "SEU_TELEGRAM_TOKEN_AQUI" or not TELEGRAM_TOKEN:
        print("⚠️ AVISO: TELEGRAM_TOKEN não configurado no arquivo .env!")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("pokemon", pokemon_command))
    app.add_handler(CommandHandler("random", random_pokemon))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot da Pokédex IA iniciado com sucesso!")
    app.run_polling()

if __name__ == "__main__":
    main()


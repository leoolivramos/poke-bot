# telegram_bot.py
import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8118490784:AAHHF6xcI1cmyhPTXcU7rSE-dcKQyM2oArI")
API_URL = os.getenv("API_URL", "https://ledevrel-ia-pokemon.hf.space/ask")

# ---------------- HANDLERS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Olá! Eu sou a Pokédex IA. Me pergunte sobre qualquer Pokémon!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    await update.message.chat.send_action(action="typing")

    try:
        response = requests.post(API_URL, json={"question": user_message}, timeout=60)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "Desculpe, não encontrei uma resposta.")
        else:
            answer = f"Erro ao consultar API ({response.status_code})."
    except Exception as e:
        answer = f"Ocorreu um erro: {e}"

    await update.message.reply_text(answer)

# ---------------- MAIN ----------------

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot da Pokédex iniciado!")
    app.run_polling()

if __name__ == "__main__":
    main()

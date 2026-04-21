"""
Telegram Bot Interface - Data Analysis Agent (CSC 446 Mini-project D)
Run with: python telegram_bot.py
"""

import os
import logging
import asyncio
from pathlib import Path

from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from agent import build_agent, PLOT_DIR

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN not set! Run: setx TELEGRAM_TOKEN your_token")

# One agent per chat session
_agents = {}

WELCOME = (
    "Hi! I am DataBot, your data analysis assistant!\n\n"
    "Send me a question about your dataset, or use:\n"
    "  /load <path> - load a CSV file\n"
    "  /suggest - get analysis ideas\n"
    "  /help - show this message\n\n"
    "Example: Load sample_data/titanic.csv"
)

HELP_TEXT = (
    "Available commands:\n"
    "  /load <path> - load a CSV dataset\n"
    "  /suggest - suggest analyses\n"
    "  /help - show this help\n\n"
    "Example questions:\n"
    "  - Describe the Age column\n"
    "  - Are there outliers in Fare?\n"
    "  - Plot a histogram of Age\n"
    "  - What is the average fare per class?"
)


def get_agent(chat_id):
    if chat_id not in _agents:
        _agents[chat_id] = build_agent()
    return _agents[chat_id]


async def _run_agent(chat_id, query):
    plots_before = set(Path(PLOT_DIR).glob("*.png"))
    agent = get_agent(chat_id)
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, agent.run, query)
    plots_after = set(Path(PLOT_DIR).glob("*.png"))
    new_plots = [str(p) for p in plots_after - plots_before]
    return str(answer), new_plots


async def _send_answer(update, text, plots):
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        await update.message.reply_text(chunk)
    for plot_path in plots:
        try:
            with open(plot_path, "rb") as f:
                await update.message.reply_photo(
                    photo=InputFile(f),
                    caption=f"Chart: {Path(plot_path).name}"
                )
        except Exception as e:
            logger.warning(f"Could not send plot: {e}")


async def cmd_start(update, context):
    await update.message.reply_text(WELCOME)


async def cmd_help(update, context):
    await update.message.reply_text(HELP_TEXT)


async def cmd_load(update, context):
    if not context.args:
        await update.message.reply_text("Usage: /load sample_data/titanic.csv")
        return
    path = " ".join(context.args)
    await update.message.reply_text(f"Loading {path}...")
    chat_id = update.effective_chat.id
    try:
        answer, plots = await _run_agent(chat_id, f"Load the dataset at {path}")
        await _send_answer(update, answer, plots)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def cmd_suggest(update, context):
    await update.message.reply_text("Thinking about what to analyse...")
    chat_id = update.effective_chat.id
    try:
        answer, plots = await _run_agent(chat_id, "Suggest analyses for the current dataset")
        await _send_answer(update, answer, plots)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


async def handle_message(update, context):
    user_text = update.message.text.strip()
    if not user_text:
        return
    chat_id = update.effective_chat.id
    await update.message.reply_text("Analysing...")
    try:
        answer, plots = await _run_agent(chat_id, user_text)
        await _send_answer(update, answer, plots)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        await update.message.reply_text(
            "Something went wrong. Try rephrasing your question "
            "or use /load to load a dataset first."
        )


def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("load", cmd_load))
    app.add_handler(CommandHandler("suggest", cmd_suggest))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("DataBot is running! Open Telegram and message your bot.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

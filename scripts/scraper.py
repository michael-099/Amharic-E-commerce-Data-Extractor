from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import json
import os
from dotenv import load_dotenv


load_dotenv()

api_id = int(os.getenv("api_id"))
api_hash = os.getenv("api_hash")
phone = os.getenv("phone")

client = TelegramClient("ecommerce_scraper", api_id, api_hash)
client.start(phone)

channels = [
    " @ZemenExpress",
    "@nevacomputer",
    "@meneshayeofficial",
    "@ethio_brand_collection",
    "@Leyueqa",
    "@sinayelj",
]


def scrape_channel(channel, limit=100):
    messages = []
    for message in client.iter_messages(channel, limit=limit):
        item = {
            "channel": channel,
            "text": message.message,
            "timestamp": str(message.date),
            "sender_id": message.sender_id,
            "media": None,
        }
        if message.photo:
            path = f"media/{channel}_{message.id}.jpg"
            message.download_media(path)
            item["media"] = path
        messages.append(item)
    return messages


os.makedirs("media", exist_ok=True)
all_data = []
for ch in channels:
    all_data.extend(scrape_channel(ch, limit=200))

with open("raw_telegram_data.jsonl", "w", encoding="utf-8") as f:
    for entry in all_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

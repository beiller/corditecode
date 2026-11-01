import asyncio
import logging
import os
import time
from typing import Callable, Optional, Tuple

import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

client = None
outbox: list[tuple[str, str, str]] = []  # (user_id, action, text) where action is "status" | "message"
inbox = []

queue_query: Callable[[str, str], None] = None
get_response: Callable[None, Tuple[str, str, str, bool]] = None
set_interrupt: Callable[[str], None] = None

STATUS_ROLES = {"tool", "error", "info"}
MAX_STATUS_LEN = 1900


def rate_limit(calls, period=10):
    def decorator(func):
        timestamps = []

        def wrapper(*args, **kwargs):
            now = time.time()
            timestamps[:] = [t for t in timestamps if now - t < period]
            if len(timestamps) >= calls:
                time.sleep(period - (now - timestamps[0]))
            timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper

    return decorator


user_id_mapping = {}
status_msg_ids: dict[str, str] = {}


@rate_limit(10, 10)
async def send_new_message(user_id: str, message: str) -> Optional[str]:
    if message == "":
        logger.warning("(discord) discarded empty message")
        return None
    global client, user_id_mapping
    channel_id, _ = user_id_mapping[user_id].split(":")
    channel = await client.fetch_channel(channel_id)
    try:
        sent = await channel.send(message)
        return str(sent.id)
    except Exception as e:
        logger.error(f"(discord) error sending message: {e}")
        return None


@rate_limit(10, 10)
async def edit_message(user_id: str, message_id: str, text: str) -> Optional[str]:
    global client
    channel_id = user_id_mapping[user_id].split(":")[0]
    channel = await client.fetch_channel(channel_id)
    try:
        original_message = await channel.fetch_message(int(message_id))
        await original_message.edit(content=text)
        return message_id
    except Exception as e:
        logger.error(f"(discord) error editing message: {e}")
        return None


async def post_status(user_id: str, text: str) -> None:
    """Update the rolling status message in place; create a new one if needed."""
    msg_id = status_msg_ids.get(user_id)
    if msg_id is not None:
        if await edit_message(user_id, msg_id, text):
            return
    # No existing status message (or edit failed): post a fresh one
    if len(text) > MAX_STATUS_LEN:
        text = text[-MAX_STATUS_LEN:]
    new_id = await send_new_message(user_id, text)
    if new_id:
        status_msg_ids[user_id] = new_id


def split_message(message: str, chunk_size: int = 1500) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    chunks = []
    start = 0
    while start < len(message):
        end = min(start + chunk_size, len(message))
        chunks.append(message[start:end])
        start = end
    return chunks


async def message_poll():
    global outbox

    while True:
        if outbox:
            for user_id, action, text in outbox:
                if client and not client.is_closed() and len(text) > 0:
                    if action == "status":
                        await post_status(user_id, text)
                    else:
                        # New message (assistant reply); reset the rolling status
                        for chunk in split_message(text, 1500):
                            await send_new_message(user_id, chunk)
                        status_msg_ids.pop(user_id, None)
            outbox = []
        await asyncio.sleep(0.1)


def get_unique_query_id(message):
    message_id = message.id
    channel_id = message.channel.id
    query_id = str(channel_id) + ":" + str(message_id)
    return query_id


async def digest_user_query():
    global inbox, user_id_mapping
    while True:
        while inbox:
            message = inbox.pop()
            query_id = get_unique_query_id(message)
            user_id_mapping[str(message.author.id)] = query_id
            await queue_query(str(message.author.id), message.content)
        await asyncio.sleep(0.01)


async def digest_agent_response():
    global outbox
    message = ""
    while True:
        user_id, role, token, chunk = await get_response()
        if role == "thinking":
            continue
        if chunk:
            message += token
        else:
            text = message + token
            message = ""
            if not text:
                continue
            if role in STATUS_ROLES:
                line = text[:200] + ("..." if len(text) > 200 else "")
                outbox.append((user_id, "status", line))
            elif role == "assistant":
                outbox.append((user_id, "message", text))


class DiscordClient(commands.Bot):
    """Custom Bot class that handles events properly"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        logger.info(f"Discord bot ready as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        logger.info(f"DM from {message.author}: {message.content}")
        if message.content.lower().strip() == "stop":
            await set_interrupt(str(message.author.id))
        else:
            inbox.append(message)
        # Optionally reply to confirm receipt
        # await message.reply("Message received!")

    async def on_raw_message_edit(self, payload):
        """Handle message edits if needed"""
        pass


async def run_discord_client():
    global client

    # Disable privileged intents to avoid the PrivilegedIntentsRequired error
    intents = discord.Intents.default()
    # Explicitly disable privileged intents
    intents.message_content = False  # Not needed for DMs
    intents.members = False  # Not used in this code

    client = DiscordClient(command_prefix="!", intents=intents)

    # Run in background
    await client.start(os.getenv("DISCORD_TOKEN"), reconnect=True)


def on_ready():
    pass


def stop():
    global client
    if client and not client.is_closed():
        if asyncio.get_event_loop().is_running():
            asyncio.run_coroutine_threadsafe(client.close(), asyncio.get_event_loop())
        else:
            asyncio.run(client.close())


async def init(
    _queue_query: Callable[[str, str], None],
    _get_response: Callable[None, Tuple[str, str, str, bool]],
    _set_interrupt: Callable[[str], None],
):
    global queue_query, get_response, set_interrupt

    queue_query = _queue_query
    get_response = _get_response
    set_interrupt = _set_interrupt

    # Run the Discord bot and I/O digesters concurrently on the event loop
    await asyncio.gather(run_discord_client(), digest_agent_response(), digest_user_query(), message_poll())

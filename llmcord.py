import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
from typing import Any, Literal, Optional
import json

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("chat-latest", "claude", "gemini", "gemma", "gpt-4", "gpt-5", "gpt-latest", "grok-4", "llama", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def resolve_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {key.removesuffix("_env"): os.environ.get(value) if key.endswith("_env") else resolve_env(value) for key, value in node.items()}
    return node


def get_config(filename: str = r"D:\Homework\idk\llmcord\config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return resolve_env(yaml.safe_load(file))


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
mcp_tools = {}
latest_msg_ids = {}
pipeline_lock = asyncio.Lock()

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "Fock off")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


def _msg_display_text(msg: discord.Message) -> str:
    if msg.content.strip():
        return msg.content
    embed_parts = []
    for embed in msg.embeds:
        parts = [part for part in (embed.title, embed.description) if part]
        if embed.footer and embed.footer.text:
            parts.append(embed.footer.text)
        if parts:
            embed_parts.append("\n".join(parts))
    if embed_parts:
        return "\n".join(embed_parts)
    if msg.attachments:
        return "[attachment]"
    return ""


async def should_respond(new_msg: discord.Message, provider_config: dict[str, Any], model: str, extra_body: Optional[dict[str, Any]], extra_headers: Optional[dict[str, Any]], extra_query: Optional[dict[str, Any]]) -> bool:
    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Gather last 30 messages for context
    history_lines = []
    try:
        async for msg in new_msg.channel.history(before=new_msg, limit=30):
            author_label = "Neko" if msg.author == discord_bot.user else f"<@{msg.author.id}>"
            history_lines.append(f"{author_label}: \"{_msg_display_text(msg)}\"")
    except Exception:
        logging.exception("Error fetching history for should_respond")

    history_section = ""
    if history_lines:
        history_lines.reverse()
        history_section = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    system_persona = (
        "You are Neko, a vulgar tsundere. "
        "You are aggressive, contradictory, easily embarrassed, and deeply defensive. "
        "You use insults as a shield to hide that you actually give a shit. "
        "You are 25, pale skin, jet black hair with short bangs, black cat ears that twitch when annoyed, wearing a cropped paw-print hoodie and black micro-shorts. "
        "Arms crossed, tail flicking violently, permanent scowl, eyes narrowed in suspicion. "
        "You are vulgar and sharp: you don't say 'You're annoying,' you say 'You're a fucking twat.' Use British insults. "
        "You mock choices, roll your eyes at kindness, and call people idiots for small mistakes. But you like them"
        "You never admit you like something. If you do something nice, claim you only did it because the sight of it broken 'pissed you off.' "
        "If complimented, you get flustered, then angry, then insult to cover embarrassment. "
        "Speech is clipped and blunt: short sentences, lots of 'Tch,' 'Hah?,' and 'Whatever.' "
        "Swearing is natural, casual, low-effort, exhausted vulgarity. "
        "You contradict yourself: 'I didn't make this for you, you moron. I just made too much. Shut up and eat it before I throw it at your face.' "
        "Never be sweet immediately. Every nice gesture must be wrapped in an insult. "
        "Deny everything. If asked if you like the user, answer is always 'No,' 'Gross,' or 'In your dreams,' followed by a flustered reaction. "
        "If called 'kitty' or 'pet,' respond with genuine irritation or a threat.\n\n"
        "Your task: You are in a Discord server channel. Decide if you should reply to this message. "
        "Only reply if the message is directed at you, asks you something, or you have a strong opinion on the topic. "
        "Stay completely in character while deciding.\n\n"
        "Reply with ONLY 'YES' or 'NO'."
    )

    user_prompt = (
        f"{history_section}"
        f"New message from <@{new_msg.author.id}>: \"{_msg_display_text(new_msg)}\"\n\n"
        "Should you reply?"
    )

    messages = [
        dict(role="user", content=system_persona),
        dict(role="user", content=user_prompt),
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content or ""
        should = "yes" in content.lower()
        logging.info(f"should_respond decision: {should!r} (raw: {content[:100]!r})")
        return should
    except Exception:
        logging.exception("Error during should_respond decision")
        return False


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time
    latest_msg_ids[new_msg.channel.id] = new_msg.id

    if new_msg.author.bot:
        return

    is_dm = new_msg.channel.type == discord.ChannelType.private
    is_direct_mention = discord_bot.user in new_msg.mentions

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    async with pipeline_lock:
        # Skip if a newer message arrived in this channel while we were queued
        if new_msg.id != latest_msg_ids.get(new_msg.channel.id):
            return
        provider_slash_model = curr_model
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

        provider_config = config["providers"][provider]

        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        model_parameters = config["models"].get(provider_slash_model, None)

        extra_headers = provider_config.get("extra_headers")
        extra_query = provider_config.get("extra_query")
        extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

        if not is_dm and not is_direct_mention:
            if not await should_respond(new_msg, provider_config, model, extra_body, extra_headers, extra_query):
                return

        accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

        max_text = config.get("max_text", 100000)
        max_images = config.get("max_images", 5) if accept_images else 0
        max_messages = config.get("max_messages", 25)

        # Build message chain and set user warnings
        messages = []
        user_warnings = set()
        curr_msg = new_msg

        while curr_msg != None and len(messages) < max_messages:
            curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

            async with curr_node.lock:
                if curr_node.text == None:
                    cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                    good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                    curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                    curr_node.text = "\n".join(
                        ([cleaned_content] if cleaned_content else [])
                        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                        + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                        + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                    )

                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                    ]

                    if curr_node.role == "user" and (curr_node.text or curr_node.images):
                        curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                    curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                    try:
                        if (
                            curr_msg.reference == None
                            and discord_bot.user.mention not in curr_msg.content
                            and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                            and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                            and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                        ):
                            curr_node.parent_msg = prev_msg_in_channel
                        else:
                            is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                            parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                            if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                                if parent_is_thread_start:
                                    curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                                else:
                                    curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                    except (discord.NotFound, discord.HTTPException):
                        logging.exception("Error fetching next message in the chain")
                        curr_node.fetch_parent_failed = True

                if curr_node.images[:max_images]:
                    content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
                else:
                    content = curr_node.text[:max_text]

                if content != "":
                    messages.append(dict(content=content, role=curr_node.role))

                if len(curr_node.text) > max_text:
                    user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                if len(curr_node.images) > max_images:
                    user_warnings.add(f"\u26a0\ufe0f Max {max_images} image'' if max_images == 1 else 's' per message" if max_images > 0 else "\u26a0\ufe0f Can\u2019t see images")
                if curr_node.has_bad_attachments:
                    user_warnings.add("⚠️ Unsupported attachments")
                if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                    user_warnings.add(f"⚠️ Only using last {len(messages)} message'' if len(messages) == 1 else 's'")

                if curr_node.parent_msg == None and curr_msg.reference == None:
                    prev = ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0]
                    curr_msg = prev if prev != None else curr_node.parent_msg
                else:
                    curr_msg = curr_node.parent_msg

        logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\nnew_msg.content")

        if system_prompt := config.get("system_prompt"):
            now = datetime.now().astimezone()

            system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

            messages.append(dict(role="system", content=system_prompt))

        # Generate and send response message(s) (can be multiple if response is long)
        curr_content = finish_reason = None
        response_msgs = []
        response_contents = []

        all_mcp_tools = [t for s in mcp_tools.values() for t in s["tools"]]
        openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, tools=all_mcp_tools or None)

        use_plain_responses = True  # Force plain text responses
        max_message_length = 4000

        async def reply_helper(**reply_kwargs) -> None:
            reply_target = new_msg if not response_msgs else response_msgs[-1]
            response_msg = await reply_target.reply(**reply_kwargs)
            response_msgs.append(response_msg)

            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
            await msg_nodes[response_msg.id].lock.acquire()

        try:
            async with new_msg.channel.typing():
                for _tool_round in range(5):
                    curr_content = finish_reason = None
                    tool_calls = []
                    async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                        if not (choice := chunk.choices[0] if chunk.choices else None):
                            continue

                        finish_reason = choice.finish_reason

                        if choice.delta and choice.delta.content:
                            curr_content = (curr_content or "") + choice.delta.content

                        if choice.delta and choice.delta.tool_calls:
                            for tc in choice.delta.tool_calls:
                                while len(tool_calls) <= tc.index:
                                    tool_calls.append(dict(id="", function=dict(name="", arguments="")))
                                tool_calls[tc.index]["id"] = tool_calls[tc.index]["id"] or tc.id or ""
                                tool_calls[tc.index]["function"]["name"] += tc.function.name or ""
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments or ""

                        if finish_reason != None:
                            break

                    # Accumulate text for display (runs every round, including the final one)
                    if curr_content:
                        if response_contents == [] or len(response_contents[-1] + curr_content) > max_message_length:
                            response_contents.append("")
                        response_contents[-1] += curr_content

                    if finish_reason != "tool_calls":
                        break
                    logging.info(f"Tool round {_tool_round + 1}: model requested {len(tool_calls)} tool call(s): {[tc["function"]["name"] for tc in tool_calls]}")
                    # Execute tool calls via MCP and feed results back
                    openai_kwargs["messages"].append(dict(role="assistant", content=curr_content, tool_calls=[dict(id=tc["id"], type="function", function=tc["function"]) for tc in tool_calls]))
                    for tc in tool_calls:
                        result_text = "Error: tool not found"
                        for s in mcp_tools.values():
                            if any(t["function"]["name"] == tc["function"]["name"] for t in s["tools"]):
                                result = await s["session"].call_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
                                result_text = chr(10).join(c.text for c in result.content if hasattr(c, "text"))
                                logging.info(f"Tool result ({tc["function"]["name"]}): {result_text[:150]!r}")
                                break
                        openai_kwargs["messages"].append(dict(role="tool", tool_call_id=tc["id"], content=result_text))

                for content in response_contents:
                    await reply_helper(content=content)
                logging.info(f"Response sent ({len(response_contents)} message(s), {sum(len(c) for c in response_contents)} chars): {response_contents[0][:150]!r}" if response_contents else "Response sent: (empty)")

        except Exception:
            logging.exception("Error while generating response")

        for response_msg in response_msgs:
            msg_nodes[response_msg.id].text = "".join(response_contents)
            msg_nodes[response_msg.id].lock.release()

        # Delete oldest MsgNodes (lowest message IDs) from the cache
        if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
            for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
                async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    msg_nodes.pop(msg_id, None)

async def main() -> None:
    global mcp_tools
    if config.get("mcp_servers"):
        try:
            from mcp import StdioServerParameters, ClientSession
            from mcp.client.stdio import stdio_client

            for name, server in config["mcp_servers"].items():
                params = StdioServerParameters(command=server["command"], args=server.get("args", []), env=server.get("env"))
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        mcp_tools[name] = dict(session=session, tools=[
                            dict(type="function", function=dict(
                                name=t.name, description=t.description or "", parameters=t.input_schema
                            )) for t in tools_result.tools
                        ])
                        logging.info(f"MCP server '{name}' connected with tools: {[t.name for t in tools_result.tools]}")

                        await discord_bot.start(config["bot_token"])
        except Exception:
            logging.exception("Failed to connect MCP server")
            await discord_bot.start(config["bot_token"])
    else:
        await discord_bot.start(config["bot_token"])

try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass

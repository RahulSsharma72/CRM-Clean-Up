"""Slack integration: python -m contact_finder.slack_app

Runs the bot over Slack Socket Mode (no public URL needed for Slack itself).
Users can:
  * @mention the bot in a channel  -> it replies in a thread and remembers the thread
  * DM the bot                      -> it replies in the DM
  * /findcontact <query>           -> it replies in the channel thread of the command

If APOLLO_WEBHOOK_URL is set, a small HTTP server also listens for Apollo's
asynchronous mobile-number webhook and posts the numbers back into the
Slack thread that asked for them.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, abort, request  # noqa: E402
from slack_bolt import App  # noqa: E402
from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

from .agent import ContactFinderAgent, Conversation  # noqa: E402
from .apollo import summarize_phone_numbers  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("contact_finder.slack")

MAX_CONVERSATIONS = 500
ALLOWED_USERS = {u.strip() for u in os.environ.get("ALLOWED_SLACK_USER_IDS", "").split(",") if u.strip()}
MENTION_RE = re.compile(r"<@[A-Z0-9]+>")

app = App(token=os.environ["SLACK_BOT_TOKEN"])
agent = ContactFinderAgent()

_convos: OrderedDict[tuple[str, str], Conversation] = OrderedDict()
_convos_lock = threading.Lock()

# apollo person id -> list of (channel, thread_ts, person name) waiting for a mobile number
_pending_phones: dict[str, list[tuple[str, str | None, str]]] = {}
_pending_lock = threading.Lock()


def get_conversation(key: tuple[str, str]) -> Conversation:
    with _convos_lock:
        convo = _convos.get(key)
        if convo is None:
            convo = _convos[key] = Conversation()
        _convos.move_to_end(key)
        while len(_convos) > MAX_CONVERSATIONS:
            _convos.popitem(last=False)
        return convo


def handle_query(client, *, user: str, text: str, channel: str, thread_ts: str | None, msg_ts: str | None) -> None:
    if ALLOWED_USERS and user not in ALLOWED_USERS:
        client.chat_postEphemeral(channel=channel, user=user, text="You're not authorised to use Contact Finder.")
        return
    text = MENTION_RE.sub("", text or "").strip()
    if not text:
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="Give me a name, company, website, LinkedIn URL or a role (e.g. _CFO at acme.com_).",
        )
        return

    if msg_ts:
        try:
            client.reactions_add(channel=channel, timestamp=msg_ts, name="mag")
        except Exception:  # reaction is cosmetic
            pass

    def on_phone_pending(people: list[dict[str, Any]]) -> None:
        with _pending_lock:
            for p in people:
                if p.get("apollo_id") and not p.get("mobile_phone"):
                    _pending_phones.setdefault(p["apollo_id"], []).append((channel, thread_ts, p.get("name", "")))

    convo = get_conversation((channel, thread_ts or "dm"))
    try:
        answer = agent.reply(convo, text, surface="slack", on_phone_pending=on_phone_pending)
    except Exception as e:  # surface failures in Slack instead of dying silently
        log.exception("Agent failed")
        answer = f":warning: Something went wrong: {e}"
    client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=answer, unfurl_links=False)


@app.event("app_mention")
def on_mention(event, client):
    thread_ts = event.get("thread_ts") or event["ts"]
    handle_query(
        client, user=event.get("user", ""), text=event.get("text", ""),
        channel=event["channel"], thread_ts=thread_ts, msg_ts=event["ts"],
    )


@app.event("message")
def on_message(event, client):
    # Only plain DMs from humans; channel messages are handled via app_mention.
    if event.get("channel_type") != "im" or event.get("subtype") or event.get("bot_id"):
        return
    handle_query(
        client, user=event.get("user", ""), text=event.get("text", ""),
        channel=event["channel"], thread_ts=event.get("thread_ts"), msg_ts=event["ts"],
    )


@app.command("/findcontact")
def on_command(ack, command, client):
    ack()
    channel = command["channel_id"]
    header = client.chat_postMessage(
        channel=channel, text=f"<@{command['user_id']}> asked: _{command.get('text', '')}_"
    )
    handle_query(
        client, user=command["user_id"], text=command.get("text", ""),
        channel=channel, thread_ts=header["ts"], msg_ts=None,
    )


# ---------------------------------------------------------------- Apollo webhook
webhook = Flask(__name__)
WEBHOOK_SECRET = os.environ.get("APOLLO_WEBHOOK_SECRET", "")


def _iter_people(payload: Any):
    """Yield every dict that looks like a person with phone numbers, wherever it is nested."""
    if isinstance(payload, dict):
        if payload.get("id") and "phone_numbers" in payload:
            yield payload
        for v in payload.values():
            yield from _iter_people(v)
    elif isinstance(payload, list):
        for v in payload:
            yield from _iter_people(v)


@webhook.post("/apollo/webhook")
def apollo_webhook():
    if WEBHOOK_SECRET and request.args.get("token") != WEBHOOK_SECRET:
        abort(403)
    payload = request.get_json(silent=True) or {}
    for person in _iter_people(payload):
        phones = summarize_phone_numbers(person.get("phone_numbers"))
        with _pending_lock:
            waiters = _pending_phones.pop(person["id"], [])
        for channel, thread_ts, name in waiters:
            if phones:
                lines = "\n".join(f"• {p['number']} ({p.get('type', 'phone')})" for p in phones)
                text = f":telephone_receiver: Phone numbers for *{name}*:\n{lines}"
            else:
                text = f"Apollo has no mobile number for *{name}*."
            app.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)
    return {"ok": True}


def start_webhook_server() -> None:
    port = int(os.environ.get("PORT", "3000"))
    server = make_server("0.0.0.0", port, webhook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("Apollo webhook listening on :%s/apollo/webhook", port)


def main() -> None:
    if os.environ.get("APOLLO_WEBHOOK_URL"):
        start_webhook_server()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()

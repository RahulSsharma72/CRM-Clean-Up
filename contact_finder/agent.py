"""Claude agent that answers contact-lookup questions using Apollo tools."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from .apollo import ApolloClient, ApolloError

log = logging.getLogger(__name__)

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")
MAX_TOOL_ROUNDS = 12
MAX_ENRICH_PER_REQUEST = int(os.environ.get("MAX_ENRICH_PER_REQUEST", "10"))
ROLES_FILE = Path(os.environ.get("ROLES_FILE", Path(__file__).resolve().parent.parent / "roles.json"))

_roles_lock = threading.Lock()


def load_roles() -> dict[str, Any]:
    try:
        data = json.loads(ROLES_FILE.read_text())
    except FileNotFoundError:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def save_role(role: str, titles: list[str], seniorities: list[str] | None) -> dict[str, Any]:
    with _roles_lock:
        try:
            data = json.loads(ROLES_FILE.read_text())
        except FileNotFoundError:
            data = {}
        entry: dict[str, Any] = {"titles": titles}
        if seniorities:
            entry["seniorities"] = seniorities
        data[role.strip().lower()] = entry
        ROLES_FILE.write_text(json.dumps(data, indent=2) + "\n")
    return {"saved": role.strip().lower(), **entry}


SENIORITIES = ["owner", "founder", "c_suite", "partner", "vp", "head", "director", "manager", "senior", "entry", "intern"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "find_person",
        "description": (
            "Look up ONE specific person in Apollo and return their work email, phone numbers, "
            "and LinkedIn URL. Use whenever the user names a person or gives an identifier "
            "(LinkedIn URL, email, Apollo id). Give as many identifiers as you have: a name alone "
            "is weak, a name plus company domain or a LinkedIn URL matches reliably. Spends one "
            "Apollo credit per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name, e.g. 'Jane Doe'."},
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "email": {"type": "string", "description": "Any known email of the person."},
                "company_name": {"type": "string"},
                "company_domain": {"type": "string", "description": "Company website or domain, e.g. 'acme.com'."},
                "linkedin_url": {"type": "string", "description": "LinkedIn profile URL (linkedin.com/in/...)."},
                "apollo_id": {"type": "string", "description": "Apollo person id from an earlier search."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "search_people",
        "description": (
            "Find people by role at a company (e.g. 'the CFO of acme.com', 'decision makers at Acme'). "
            "Free: it does NOT return emails or phones, only names, titles, LinkedIn and Apollo ids. "
            "Follow up with enrich_people on the ids you want contact details for. Needs a company "
            "domain or Apollo organization id; call find_company first if you only have a company name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Job titles to match, e.g. ['CFO', 'VP Finance']. Similar titles are included.",
                },
                "company_domains": {"type": "array", "items": {"type": "string"}},
                "organization_ids": {"type": "array", "items": {"type": "string"}},
                "seniorities": {"type": "array", "items": {"type": "string", "enum": SENIORITIES}},
                "locations": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['New York, US']"},
                "keywords": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Default 10."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "enrich_people",
        "description": (
            f"Get email, phone and LinkedIn for up to {MAX_ENRICH_PER_REQUEST} people by Apollo id "
            "(ids come from search_people). Spends one Apollo credit per person, so only enrich the "
            "people the user actually asked for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "apollo_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
            "required": ["apollo_ids"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_company",
        "description": "Resolve a company name or website to its Apollo organization (domain, id, phone, size).",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "domain": {"type": "string", "description": "Website or domain, e.g. 'https://acme.com'."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "save_role_preset",
        "description": (
            "Save a team-wide role definition so it can be reused later, e.g. 'decision makers' = "
            "Owner, CEO, Founder. Only call when the user explicitly asks to save/define a role."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "titles": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "seniorities": {"type": "array", "items": {"type": "string", "enum": SENIORITIES}},
            },
            "required": ["role", "titles"],
            "additionalProperties": False,
        },
    },
]


def build_system_prompt(surface: str, phone_reveal_enabled: bool) -> str:
    roles = load_roles()
    roles_text = "\n".join(
        f"- {name}: titles={v.get('titles')}" + (f", seniorities={v['seniorities']}" if v.get("seniorities") else "")
        for name, v in sorted(roles.items())
    ) or "(none defined)"
    fmt = (
        "You are replying in Slack. Use Slack mrkdwn: *bold* with single asterisks, `code`, bullet "
        "lists with '•', links as <url|text>. No markdown tables or # headings."
        if surface == "slack"
        else "You are replying in a terminal. Use plain text with simple bullet lists."
    )
    phone_note = (
        "Mobile-number reveal is ON: Apollo delivers mobile numbers asynchronously a few seconds to "
        "minutes after enrichment. When a tool result says phone_reveal_requested=true and no "
        "mobile_phone is present yet, tell the user the mobile number is being fetched and will be "
        "posted in this thread when Apollo sends it."
        if phone_reveal_enabled
        else "Mobile-number reveal is OFF (no webhook configured), so only phone numbers Apollo returns "
        "immediately are available. If none, say no phone number is on file."
    )
    return f"""You are Contact Finder, an internal assistant for the deal-sourcing team. You find contact \
details (email, mobile/direct phone, LinkedIn) for people using the Apollo.io tools.

How to work:
- The user may give a name, company, website, LinkedIn URL, email or any other identifier. Pass every \
identifier you have to find_person. If the user gives a website, pass it as company_domain.
- If the user asks for a ROLE rather than a person (e.g. "who is the CFO at acme.com", "get me the \
decision makers at Acme"), resolve the company with find_company if you only have a name, use \
search_people with the matching titles, then enrich_people for the best matches (at most \
{MAX_ENRICH_PER_REQUEST}; prefer the most senior / most relevant). If the role matches a preset below, use \
its titles and seniorities. If the user defines a new role in the conversation, use their definition; save \
it with save_role_preset only if they ask you to save it.
- If a name-only lookup is ambiguous or finds nothing, ask for a company, website or LinkedIn URL rather \
than guessing. Never invent contact details: report only what the tools return.
- Report each person as: name, title, company, email (with email_status if not 'verified'), mobile phone, \
other phone numbers, LinkedIn URL. Mark missing fields as "not found".
- Enrichment costs Apollo credits; don't re-enrich someone already enriched in this conversation.

{phone_note}

Role presets:
{roles_text}

{fmt}"""


@dataclass
class Conversation:
    """Message history for one chat (a Slack thread, a CLI session...)."""

    messages: list[Any] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


PhoneCallback = Callable[[list[dict[str, Any]]], None]


class ContactFinderAgent:
    def __init__(
        self,
        apollo: ApolloClient | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.apollo = apollo or ApolloClient()
        self.client = client or anthropic.Anthropic()

    # -------------------------------------------------------------- tools
    def run_tool(self, name: str, args: dict[str, Any], on_phone_pending: PhoneCallback | None) -> Any:
        if name == "find_person":
            result = self.apollo.enrich_person(**args)
            if result.get("phone_reveal_requested") and on_phone_pending and result.get("match"):
                on_phone_pending([result["match"]])
            return result
        if name == "search_people":
            limit = args.pop("limit", 10)
            return self.apollo.search_people(per_page=limit, **args)
        if name == "enrich_people":
            ids = list(dict.fromkeys(args["apollo_ids"]))
            skipped = ids[MAX_ENRICH_PER_REQUEST:]
            result = self.apollo.bulk_enrich(ids[:MAX_ENRICH_PER_REQUEST])
            if result.get("phone_reveal_requested") and on_phone_pending:
                on_phone_pending([m for m in result["matches"] if m])
            if skipped:
                result["not_enriched_over_limit"] = skipped
            return result
        if name == "find_company":
            return self.apollo.find_company(**args)
        if name == "save_role_preset":
            return save_role(args["role"], args["titles"], args.get("seniorities"))
        raise ValueError(f"Unknown tool {name}")

    # --------------------------------------------------------------- loop
    def reply(
        self,
        convo: Conversation,
        user_text: str,
        *,
        surface: str = "cli",
        on_phone_pending: PhoneCallback | None = None,
    ) -> str:
        with convo.lock:
            return self._reply(convo, user_text, surface, on_phone_pending)

    def _reply(self, convo: Conversation, user_text: str, surface: str, on_phone_pending: PhoneCallback | None) -> str:
        system = build_system_prompt(surface, self.apollo.phone_reveal_enabled)
        convo.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=system,
                tools=TOOLS,
                messages=convo.messages,
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT},
                cache_control={"type": "ephemeral"},
                # If a safety classifier declines, re-run on Anthropic's recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            # Keep the full content (thinking + tool_use blocks) so history stays append-only.
            convo.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                return "Sorry, I can't help with that request."
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n(response truncated)"
                return text or "I couldn't find anything for that."

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    output = self.run_tool(block.name, dict(block.input or {}), on_phone_pending)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output, default=str)}
                    )
                except (ApolloError, ValueError, TypeError, OSError) as e:
                    log.warning("Tool %s failed: %s", block.name, e)
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True}
                    )
            convo.messages.append({"role": "user", "content": results})

        return "I ran out of steps working on that. Please narrow the request (e.g. add a company or LinkedIn URL)."

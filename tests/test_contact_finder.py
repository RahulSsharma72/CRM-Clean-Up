"""Offline tests: Apollo HTTP and the Claude API are both mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contact_finder import agent as agent_mod
from contact_finder.agent import ContactFinderAgent, Conversation
from contact_finder.apollo import ApolloClient, normalize_domain, normalize_linkedin_url

PERSON = {
    "id": "p1",
    "first_name": "Jane",
    "last_name": "Doe",
    "name": "Jane Doe",
    "title": "CFO",
    "email": "jane@acme.com",
    "email_status": "verified",
    "linkedin_url": "http://www.linkedin.com/in/janedoe",
    "phone_numbers": [{"sanitized_number": "+15551234567", "type_cd": "mobile"}],
    "organization": {"name": "Acme", "primary_domain": "acme.com"},
}


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.content = json.dumps(data).encode()
        self.text = self.content.decode()

    def json(self):
        return self._data


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, headers=None, json=None, params=None, timeout=None):
        path = url.split("/api/v1/")[1]
        self.calls.append((method, path, json, params, headers))
        return FakeResp(self.routes[path])


def make_apollo(routes, webhook_url=None):
    http = FakeHTTP(routes)
    return ApolloClient("k", webhook_url=webhook_url, session=http, reveal_personal_emails=False), http


def test_normalizers():
    assert normalize_domain("https://www.Acme.com/about") == "acme.com"
    assert normalize_domain("acme.com") == "acme.com"
    assert normalize_linkedin_url("linkedin.com/in/janedoe/") == "https://www.linkedin.com/in/janedoe"


def test_enrich_person_summarizes_and_sends_identifiers():
    apollo, http = make_apollo({"people/match": {"person": PERSON}})
    out = apollo.enrich_person(name="Jane Doe", company_domain="https://acme.com", linkedin_url="linkedin.com/in/janedoe")
    m = out["match"]
    assert m["email"] == "jane@acme.com"
    assert m["mobile_phone"] == "+15551234567"
    assert m["linkedin_url"].endswith("/in/janedoe")
    _, path, body, _, headers = http.calls[0]
    assert headers["x-api-key"] == "k"
    assert body["domain"] == "acme.com"
    assert "reveal_phone_number" not in body  # no webhook configured
    assert out["phone_reveal_requested"] is False


def test_phone_reveal_uses_webhook():
    apollo, http = make_apollo({"people/match": {"person": PERSON}}, webhook_url="https://x/apollo/webhook")
    out = apollo.enrich_person(email="jane@acme.com")
    body = http.calls[0][2]
    assert body["reveal_phone_number"] is True and body["webhook_url"] == "https://x/apollo/webhook"
    assert out["phone_reveal_requested"] is True


def test_enrich_requires_identifier():
    apollo, _ = make_apollo({})
    with pytest.raises(ValueError):
        apollo.enrich_person()


def test_search_people_payload():
    apollo, http = make_apollo({"mixed_people/api_search": {"people": [{"id": "p1", "first_name": "Jane", "title": "CFO", "has_email": True}], "total_entries": 1}})
    out = apollo.search_people(titles=["CFO"], company_domains=["www.acme.com"], per_page=50)
    body = http.calls[0][2]
    assert body["person_titles"] == ["CFO"]
    assert body["q_organization_domains_list"] == ["acme.com"]
    assert body["per_page"] == 25
    assert out["people"][0]["apollo_id"] == "p1"


# ------------------------------------------------------------------ agent loop
def block(**kw):
    return SimpleNamespace(**kw)


class FakeClaude:
    """Returns a scripted sequence of responses and records requests."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def test_agent_role_lookup_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "ROLES_FILE", tmp_path / "roles.json")
    apollo, http = make_apollo(
        {
            "mixed_people/api_search": {"people": [{"id": "p1", "first_name": "Jane", "title": "CFO"}]},
            "people/bulk_match": {"matches": [PERSON]},
        },
        webhook_url="https://x/hook",
    )
    claude = FakeClaude(
        [
            SimpleNamespace(stop_reason="tool_use", content=[block(type="tool_use", id="t1", name="search_people", input={"titles": ["CFO"], "company_domains": ["acme.com"], "limit": 5})]),
            SimpleNamespace(stop_reason="tool_use", content=[block(type="tool_use", id="t2", name="enrich_people", input={"apollo_ids": ["p1"]})]),
            SimpleNamespace(stop_reason="end_turn", content=[block(type="text", text="Jane Doe, CFO: jane@acme.com")]),
        ]
    )
    pending = []
    bot = ContactFinderAgent(apollo=apollo, client=claude)
    convo = Conversation()
    answer = bot.reply(convo, "Who is the CFO at acme.com?", on_phone_pending=pending.extend)

    assert "jane@acme.com" in answer
    assert [c[1] for c in http.calls] == ["mixed_people/api_search", "people/bulk_match"]
    assert http.calls[0][2]["per_page"] == 5
    assert pending and pending[0]["apollo_id"] == "p1"
    # tool results were fed back to Claude
    tool_result = claude.requests[2]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t2" and "jane@acme.com" in tool_result["content"]
    assert claude.requests[0]["fallbacks"] == "default"


def test_agent_reports_tool_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "ROLES_FILE", tmp_path / "roles.json")
    apollo, _ = make_apollo({})
    claude = FakeClaude(
        [
            SimpleNamespace(stop_reason="tool_use", content=[block(type="tool_use", id="t1", name="find_person", input={})]),
            SimpleNamespace(stop_reason="end_turn", content=[block(type="text", text="Need more info")]),
        ]
    )
    bot = ContactFinderAgent(apollo=apollo, client=claude)
    assert bot.reply(Conversation(), "find him") == "Need more info"
    result = claude.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True


def test_save_role_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "ROLES_FILE", tmp_path / "roles.json")
    agent_mod.save_role("Deal Team", ["Managing Partner", "Principal"], ["partner"])
    assert agent_mod.load_roles()["deal team"]["titles"] == ["Managing Partner", "Principal"]
    assert "deal team" in agent_mod.build_system_prompt("slack", False)

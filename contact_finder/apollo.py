"""Thin client for the Apollo.io REST API.

Only the endpoints the chatbot needs are wrapped:

* People Enrichment      POST /api/v1/people/match           (costs credits)
* Bulk People Enrichment POST /api/v1/people/bulk_match      (costs credits, max 10)
* People API Search      POST /api/v1/mixed_people/api_search (no credits, no contact data)
* Organization Enrich    GET  /api/v1/organizations/enrich
* Organization Search    POST /api/v1/mixed_companies/search

Every public method returns plain dicts that have been trimmed down to the
fields the chatbot cares about (see ``summarize_person``), so the model never
sees the full, very large Apollo payload.

Mobile numbers: Apollo only releases mobile/direct numbers asynchronously.
When ``reveal_phone_number`` is set, Apollo requires a ``webhook_url`` and
POSTs the number there a few seconds to minutes later. If no webhook URL is
configured we only return phone numbers Apollo already has in the synchronous
response (e.g. contacts already saved in your Apollo account, company phone).
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.apollo.io/api/v1"


class ApolloError(RuntimeError):
    """Raised when Apollo returns an error response."""


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None / empty so we don't send noise to Apollo."""
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def normalize_domain(value: str | None) -> str | None:
    """Turn 'https://www.acme.com/about' into 'acme.com'."""
    if not value:
        return None
    value = value.strip()
    if "://" not in value:
        value = "http://" + value
    host = urlparse(value).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def normalize_linkedin_url(value: str | None) -> str | None:
    """Canonicalise a LinkedIn profile URL (Apollo matches on the /in/ slug)."""
    if not value:
        return None
    value = value.strip()
    if "://" not in value:
        value = "https://" + value
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    return f"https://www.linkedin.com{path}" if "linkedin.com" in (parsed.hostname or "") else value


def summarize_phone_numbers(numbers: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for n in numbers or []:
        number = n.get("sanitized_number") or n.get("raw_number")
        if not number:
            continue
        out.append(
            _clean(
                {
                    "number": number,
                    "type": n.get("type_cd") or n.get("type"),  # mobile / work_direct / work_hq / other
                    "status": n.get("status_cd") or n.get("status"),
                }
            )
        )
    return out


def summarize_person(p: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the fields the chatbot needs from an Apollo person object."""
    if not p:
        return None
    org = p.get("organization") or {}
    phones = summarize_phone_numbers(p.get("phone_numbers"))
    # Contacts saved in Apollo sometimes carry a single sanitized_phone instead.
    if not phones and p.get("sanitized_phone"):
        phones = [{"number": p["sanitized_phone"]}]
    mobile = next((x["number"] for x in phones if x.get("type") == "mobile"), None)
    return _clean(
        {
            "apollo_id": p.get("id"),
            "name": p.get("name") or " ".join(filter(None, [p.get("first_name"), p.get("last_name")])),
            "title": p.get("title"),
            "headline": p.get("headline"),
            "seniority": p.get("seniority"),
            "location": ", ".join(filter(None, [p.get("city"), p.get("state"), p.get("country")])),
            "email": p.get("email"),
            "email_status": p.get("email_status"),
            "personal_emails": p.get("personal_emails"),
            "linkedin_url": p.get("linkedin_url"),
            "mobile_phone": mobile,
            "phone_numbers": phones,
            "company": _clean(
                {
                    "name": org.get("name") or p.get("organization_name"),
                    "domain": org.get("primary_domain"),
                    "website": org.get("website_url"),
                    "company_phone": (org.get("primary_phone") or {}).get("sanitized_number")
                    or org.get("phone"),
                }
            ),
            # Search results (no credits spent) only tell us whether data exists.
            "has_email": p.get("has_email"),
            "has_direct_phone": p.get("has_direct_phone"),
        }
    )


def summarize_org(o: dict[str, Any] | None) -> dict[str, Any] | None:
    if not o:
        return None
    return _clean(
        {
            "apollo_org_id": o.get("id"),
            "name": o.get("name"),
            "domain": o.get("primary_domain"),
            "website": o.get("website_url"),
            "linkedin_url": o.get("linkedin_url"),
            "phone": (o.get("primary_phone") or {}).get("sanitized_number") or o.get("phone"),
            "industry": o.get("industry"),
            "employees": o.get("estimated_num_employees"),
            "location": ", ".join(filter(None, [o.get("city"), o.get("state"), o.get("country")])),
        }
    )


class ApolloClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        webhook_url: str | None = None,
        reveal_personal_emails: bool | None = None,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("APOLLO_API_KEY")
        if not self.api_key:
            raise ValueError("APOLLO_API_KEY is not set")
        self.base_url = (base_url or os.environ.get("APOLLO_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.webhook_url = webhook_url if webhook_url is not None else os.environ.get("APOLLO_WEBHOOK_URL")
        if reveal_personal_emails is None:
            reveal_personal_emails = os.environ.get("APOLLO_REVEAL_PERSONAL_EMAILS", "false").lower() == "true"
        self.reveal_personal_emails = reveal_personal_emails
        self.timeout = timeout
        self.http = session or requests.Session()

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "accept": "application/json",
        }
        log.debug("Apollo %s %s params=%s body=%s", method, path, params, json)
        resp = self.http.request(method, url, headers=headers, json=json, params=params, timeout=self.timeout)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text[:500]
            raise ApolloError(f"Apollo {method} {path} failed ({resp.status_code}): {detail}")
        return resp.json() if resp.content else {}

    def _reveal_flags(self, reveal_phone: bool) -> dict[str, Any]:
        flags: dict[str, Any] = {"reveal_personal_emails": self.reveal_personal_emails}
        if reveal_phone and self.webhook_url:
            flags["reveal_phone_number"] = True
            flags["webhook_url"] = self.webhook_url
        return flags

    @property
    def phone_reveal_enabled(self) -> bool:
        return bool(self.webhook_url)

    # ---------------------------------------------------------------- people
    def enrich_person(
        self,
        *,
        name: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        email: str | None = None,
        company_name: str | None = None,
        company_domain: str | None = None,
        linkedin_url: str | None = None,
        apollo_id: str | None = None,
        reveal_phone: bool = True,
    ) -> dict[str, Any]:
        """Match one person and return their contact details (spends credits)."""
        body = _clean(
            {
                "id": apollo_id,
                "name": name,
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "organization_name": company_name,
                "domain": normalize_domain(company_domain),
                "linkedin_url": normalize_linkedin_url(linkedin_url),
            }
        )
        if not body:
            raise ValueError("Provide at least one identifier (name, email, LinkedIn URL, Apollo id...)")
        body.update(self._reveal_flags(reveal_phone))
        data = self._request("POST", "people/match", json=body)
        person = summarize_person(data.get("person"))
        return {
            "match": person,
            "phone_reveal_requested": bool(body.get("reveal_phone_number")) and bool(person),
        }

    def bulk_enrich(self, apollo_ids: list[str], *, reveal_phone: bool = True) -> dict[str, Any]:
        """Enrich up to 10 people by Apollo id (spends credits)."""
        ids = [i for i in apollo_ids if i][:10]
        if not ids:
            return {"matches": []}
        body: dict[str, Any] = {"details": [{"id": i} for i in ids]}
        body.update(self._reveal_flags(reveal_phone))
        data = self._request("POST", "people/bulk_match", json=body)
        matches = [summarize_person(m) for m in data.get("matches") or [] if m]
        return {
            "matches": matches,
            "phone_reveal_requested": bool(body.get("reveal_phone_number")) and bool(matches),
        }

    def search_people(
        self,
        *,
        titles: list[str] | None = None,
        company_domains: list[str] | None = None,
        organization_ids: list[str] | None = None,
        seniorities: list[str] | None = None,
        locations: list[str] | None = None,
        keywords: str | None = None,
        include_similar_titles: bool = True,
        per_page: int = 10,
        page: int = 1,
    ) -> dict[str, Any]:
        """Find people by role / company. Free, but returns no emails or phones."""
        body = _clean(
            {
                "person_titles": titles,
                "include_similar_titles": include_similar_titles if titles else None,
                "q_organization_domains_list": [d for d in (normalize_domain(x) for x in company_domains or []) if d],
                "organization_ids": organization_ids,
                "person_seniorities": seniorities,
                "person_locations": locations,
                "q_keywords": keywords,
                "page": page,
                "per_page": max(1, min(per_page, 25)),
            }
        )
        data = self._request("POST", "mixed_people/api_search", json=body)
        people = [summarize_person(p) for p in data.get("people") or []]
        total = data.get("total_entries")
        if total is None:
            total = (data.get("pagination") or {}).get("total_entries")
        return {"total_entries": total, "people": people}

    # ---------------------------------------------------------- organizations
    def find_company(self, *, domain: str | None = None, name: str | None = None) -> dict[str, Any]:
        domain = normalize_domain(domain)
        if domain:
            data = self._request("GET", "organizations/enrich", params={"domain": domain})
            return {"companies": [c for c in [summarize_org(data.get("organization"))] if c]}
        if not name:
            raise ValueError("Provide a company name or domain")
        data = self._request("POST", "mixed_companies/search", json={"q_organization_name": name, "per_page": 5})
        orgs = data.get("organizations") or data.get("accounts") or []
        return {"companies": [summarize_org(o) for o in orgs]}

# Contact Finder: Apollo + Claude chatbot (CLI and Slack)

Ask in plain English, get back **email, mobile phone and LinkedIn**:

```
you> Jane Doe at acme.com
you> linkedin.com/in/janedoe
you> who is the CFO at https://acme.com?
you> get me the decision makers at Acme Dental Group
you> "deal team" means Managing Partner, Principal, VP Corp Dev. Save that, then find the deal team at acme.com
```

Claude (`claude-opus-5-5`) reads the request and decides which Apollo calls to make:

| Tool | Apollo endpoint | Credits |
|---|---|---|
| `find_person`: name / email / LinkedIn / company / website → one person | `POST /people/match` | 1 per match |
| `search_people`: role + company → list of people (no contact data) | `POST /mixed_people/api_search` | free |
| `enrich_people`: contact data for up to 10 people from a search | `POST /people/bulk_match` | 1 per person |
| `find_company`: company name → Apollo org id (website → full company details) | `POST /organizations/search` (fuzzy lookup), `GET /organizations/enrich` | name: free · domain: 1 |
| `save_role_preset`: store a team-defined role in `roles.json` | — | — |

Role presets such as "decision makers", "finance" and "operations" live in `roles.json`. Edit that file, or define and save roles from chat.

## Two ways to run it in Slack

1. **No code: Claude in Slack + the Apollo connector.** On a Claude Team or Enterprise plan, an owner can set up Claude in Slack. Each person who has added the Apollo connector to their own claude.ai account can then @mention Claude or DM it, e.g. "find the email and mobile for the CFO at acme.com". Claude uses that person's own Apollo access and asks them before first use. Other people in a channel can't use your Apollo connection. See https://claude.com/docs/claude-tag/overview.md.
2. **This repo: a self-hosted bot.** It uses one shared Apollo API key for the whole team, so nobody needs their own connector. It also has role presets (`roles.json`) and a cap on credit use. The rest of this README covers this option.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and APOLLO_API_KEY
```

The Apollo key must be a **master API key**: enrichment and search endpoints reject standard keys.

### Terminal chat

```bash
python -m contact_finder.cli
```

### Slack bot

1. Go to https://api.slack.com/apps → **Create New App** → **From a manifest** and paste `slack_manifest.yml`.
2. Under **Basic Information → App-Level Tokens**, create a token with the `connections:write` scope. That is `SLACK_APP_TOKEN` (`xapp-…`).
3. Install the app to your workspace. Copy the **Bot User OAuth Token** into `SLACK_BOT_TOKEN` (`xoxb-…`).
4. Run `python -m contact_finder.slack_app` on any always-on host (a VM, Render, Fly.io, Railway…). It uses Socket Mode, so Slack needs no public URL.

To use it, `@Contact Finder Jane Doe at acme.com` in a channel (follow-ups go in the same thread), DM the bot, or run `/findcontact CFO at acme.com`.
Set `ALLOWED_SLACK_USER_IDS` to limit who can spend Apollo credits.

### Mobile numbers (optional)

Apollo does **not** return mobile numbers in the API response. It sends them to a webhook later, usually within a few minutes. To turn this on:

1. Expose the Slack bot's `PORT` publicly (your host's URL, or `ngrok http 3000` for testing).
2. Set `APOLLO_WEBHOOK_SECRET` to a random string, and
   `APOLLO_WEBHOOK_URL=https://<public-host>/apollo/webhook?token=<secret>`.

When Apollo delivers a number, the bot posts it in the Slack thread that asked for it. Mobile reveals use extra Apollo credits.
Without a webhook, the bot returns only the phone numbers Apollo gives back right away: numbers for contacts already in your Apollo account, and the company phone.

## Tests

```bash
python -m pytest -q
```

The tests mock both Apollo and Claude, so they need no keys.

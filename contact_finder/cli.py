"""Terminal chat: python -m contact_finder.cli"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
    from .agent import ContactFinderAgent, Conversation

    agent = ContactFinderAgent()
    convo = Conversation()
    print("Contact Finder. Give a name, company, website, LinkedIn URL or a role. Ctrl-D to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        print(f"\nbot> {agent.reply(convo, text, surface='cli')}\n")


if __name__ == "__main__":
    main()

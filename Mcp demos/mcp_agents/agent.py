"""
Demo MCP client — a Gemini agent using the local Order Pricing server.

    export GEMINI_API_KEY=your_key_here
    python agent.py
"""
import asyncio
import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# --- config ---------------------------------------------------------------
# Never hardcode this. Read it from the environment so it cannot end up on a
# projector, in a screenshot, or in a git commit.
API_KEY = ""

# Primary, then fallback. Both must be real model IDs — a wrong name fails
# with a 404 that no amount of retrying will fix.
MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

MAX_ATTEMPTS = 3

# stdio launches the server as a subprocess, so the path must not depend on
# the directory you happen to run this from.
SERVER = Path(__file__).parent.resolve() / "server.py"

RULE = "\u2500" * 68


def make_llm(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=API_KEY,
        # Zero for a calculation demo. You want the same answer every time
        # you run this in front of people.
        temperature=0,
        max_retries=2,
    )


def is_transient(error_text):
    """Worth retrying the same model, versus worth giving up on it."""
    markers = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
               "500", "INTERNAL", "DEADLINE_EXCEEDED")
    return any(m in error_text for m in markers)


def print_trace(messages):
    """Show the tool-calling loop, in order, so the audience can follow it."""
    print(f"\n{RULE}\n  Agent execution trace\n{RULE}")
    step = 0
    for message in messages:
        kind = getattr(message, "type", None)

        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                step += 1
                print(f"\n  [{step}] Model decided to call: {call['name']}")
                print(f"      Arguments: {call['args']}")

        elif kind == "tool":
            step += 1
            name = getattr(message, "name", "tool")
            print(f"\n  [{step}] Server returned from {name}:")
            print(f"      {message.content}")

        elif kind == "ai" and (message.content or "").strip():
            step += 1
            preview = str(message.content).strip()
            print(f"\n  [{step}] Model wrote text "
                  f"({len(preview)} chars)")


async def run_with_model(model_name, tools, question):
    """One model, up to MAX_ATTEMPTS tries. Returns True if it answered."""
    llm = make_llm(model_name)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a pricing assistant for an office supplies store. "
            "Prices are not public, so you must call lookup_item before "
            "you can calculate anything. Never invent a price. "
            "After the tools return, explain the arithmetic step by step "
            "so a human can check it."
        ),
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"    attempt {attempt}/{MAX_ATTEMPTS}")
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            print_trace(result["messages"])
            print(f"\n{RULE}\n  Final answer\n{RULE}\n")
            print(result["messages"][-1].content)
            return True

        except Exception as e:
            text = str(e)
            print(f"    failed: {text[:300]}")

            if is_transient(text) and attempt < MAX_ATTEMPTS:
                wait = attempt * 2
                print(f"    transient error, retrying in {wait}s")
                await asyncio.sleep(wait)
                continue

            # Anything else (bad model name, bad key, bad request) will not
            # improve on retry. Give up on THIS model and let the caller try
            # the next one -- do not raise, or the fallback never runs.
            return False

    return False


async def main():
    if not API_KEY:
        sys.exit("GEMINI_API_KEY is not set.\n"
                 "  macOS/Linux:  export GEMINI_API_KEY=your_key\n"
                 "  Windows:      set GEMINI_API_KEY=your_key")

    if not SERVER.exists():
        sys.exit(f"Cannot find the MCP server at {SERVER}")

    client = MultiServerMCPClient({
        "pricing": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER)],
        }
    })

    tools = await client.get_tools()

    print(f"{RULE}\n  Tools discovered over MCP\n{RULE}")
    for tool in tools:
        first_line = (tool.description or "").strip().splitlines()[0]
        print(f"\n  {tool.name}")
        print(f"    {first_line}")

    question = (
        "I need 3 laptop stands and 2 usb-c hubs. "
        "What is the total cost including GST?"
    )
    print(f"\n{RULE}\n  User\n{RULE}\n\n  {question}")

    for model_name in MODELS:
        print(f"\n{RULE}\n  Model: {model_name}\n{RULE}")
        if await run_with_model(model_name, tools, question):
            return
        print(f"  Giving up on {model_name}")

    print("\n  Every model failed. Check GEMINI_API_KEY and the model IDs "
          "in MODELS.")


if __name__ == "__main__":
    asyncio.run(main())

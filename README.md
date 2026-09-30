# Large Language Models & the Model Context Protocol

Code and slides from a technical seminar for final-year CSE students.

**Goutham Nivass** · Sr AI Engineer, Cognizant

Everything here runs locally. Nothing needs a paid API key except the Gemini
agent in `mcp-demo/`, and that has a free tier.

---

## What's in here

```
slides/                        The deck, as PDF and PPTX
llm-demos/                     Streamlit app — five interactive demos
  app.py                       All five pages
  demo.py                      Launcher (run this)
  setup.py                     One-time model download
  sampling-levers.html         Standalone version of the sampling demo
notebook/
  LLM_demos.ipynb              Same demos as a notebook, for reading the code
mcp-demo/
  server.py                    An MCP server — two tools, no framework
  agent.py                     A Gemini agent that talks to it
  claude_desktop_config.json   Template for Claude Desktop
```

The Streamlit app and the notebook cover the same ground. Use the app to
*play*, the notebook to *read the code*.

---

## Quick start

### The LLM demos

```bash
cd llm-demos
pip install -r requirements.txt
python setup.py        # once, on wifi — downloads BERT and GPT-2 into ./_models
python demo.py         # opens in your browser
```

`setup.py` pulls a few hundred MB into a local `_models/` folder. After that
the app never touches the network. To prove it, turn your wifi off and run
`python setup.py --verify`.

If you only want the tokenizer page, `pip install tiktoken streamlit` is
enough and skips the torch download.

### The MCP demo

```bash
cd mcp-demo
pip install -r requirements.txt
export GEMINI_API_KEY=your_own_key      # Google AI Studio, free tier is fine
python agent.py
```

For the Claude Desktop half, see [`mcp-demo/README.md`](mcp-demo/README.md).

---

## Slide → demo map

| Slides | Topic | Where to run it |
|---|---|---|
| 10, 11 | Tokenization, the token tax | app page 1 |
| 13, 14, 15 | Embeddings, contextual meaning, attention | app page 2 |
| 16, 17 | Next-token prediction | app page 3 |
| 18 | Prompt framing | app page 5 |
| 19 | Temperature, top-k, top-p | app page 4, or `sampling-levers.html` |
| 20–26 | MCP | `mcp-demo/` |

---

## The five LLM demos

**1 — Tokenization.** Type anything and watch it split into tokens with their
integer IDs, through two different tokenizers side by side. Includes a token
count across English, Tamil and Hindi. The same paragraph costs several times
more tokens in an Indic script — that's an API bill and a context-window
constraint, not a curiosity.

**2 — Meaning moves.** Two anchor sentences define a meaning axis. Feed it any
sentence containing the same word and see where the vector lands. The
layer-by-layer chart is the point: every line starts near zero, because layer
0 is a dictionary lookup and identical for all of them. Then attention pulls
them apart. Try "the blood bank called" or "the plane banked hard".

**3 — Next token, live.** Real GPT-2. The actual top-8 candidates with
probabilities; click one to append it and watch the sentence build. Hit "take
the top token" repeatedly and it falls into a repetition loop, which is why
nobody ships greedy decoding for prose.

**4 — Sampling levers.** Temperature, top-k and top-p over a fixed
distribution. Pure arithmetic, no model needed. The thing to notice: set top-k
to 3 and the pool stays at 3 whatever happens; set top-p to 0.9 and the pool
resizes itself as the distribution changes shape.

**5 — Prompt framing.** Zero-shot versus few-shot, measured as *how much
probability mass lands on the answers you actually wanted*. Then change the
wanted answers from `positive, negative` to `good, bad` and watch the mass
collapse — the examples taught a vocabulary, not a concept.

### A note on the models

These use **GPT-2 (2019)** and **BERT-base (2018)**. Both are small and old,
deliberately. They run on a laptop CPU with no API key, and the mechanisms —
tokenization, contextual embeddings, next-token sampling — are identical in a
frontier model. Watching them work imperfectly makes the mechanism more
visible, not less.

---

## The MCP demo

An MCP server with two tools:

- `lookup_item(item_name)` — returns a price the model cannot possibly guess
- `calculate_total(quantity, unit_price, gst_percent)` — does the arithmetic

Ask for two different items and the model works out on its own that it needs
the lookups before the calculations. Four tool calls, in an order nobody
specified. That composition is the part worth understanding.

Then run the **same `server.py`** two ways:

1. Through `agent.py`, which uses Gemini via LangChain
2. As a connector in Claude Desktop

Zero code changes between them. Different vendor, different model, different
host application. That's the M+N argument in practice rather than on a slide.

### Worth trying

Change `lookup_item`'s docstring to something vague like `"gets item stuff"`,
restart, ask the same question. It will pick the wrong tool or ask you what
you meant. **Tool descriptions are prompts.** Writing them badly is the most
common reason MCP servers don't work, and it's the cheapest thing to fix.

---

## If you want to build something

Ranked by how much you'll learn, not by how impressive the demo looks:

1. **An MCP server for your own college.** Timetable, attendance, exam
   schedule. Read-only first. Genuinely useful, and nobody has built it.
2. **An eval harness.** Twenty questions, the tool-call sequence you expect
   for each, and a script that scores them. Rarest skill in the room.
3. **A server that fails loudly.** Take a working one, write deliberately bad
   tool descriptions, document exactly what breaks. Almost nobody writes this
   up, and it proves you understand the mechanism rather than the tutorial.
4. **Two tools that must chain.** The moment a model composes your tools
   without being told how is the moment agents stop being magic.

A project with a written-up failure analysis beats a project with a working
demo. Interviewers can tell the difference.

---

## Further reading

- [modelcontextprotocol.io](https://modelcontextprotocol.io) — the spec. Short
  and readable; the transport section is worth reading properly
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et
  al., 2017. Note it describes an encoder-decoder built for translation;
  GPT-style models are decoder-only, which is half that famous diagram
  repeated
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) —
  live GPT-2 in the browser
- [bbycroft.net/llm](https://bbycroft.net/llm) — a 3D walkthrough of a tiny
  model. The green/blue split between computed values and learned weights is
  the most valuable thing on it
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  — Jay Alammar

---

## Gotchas

**Streamlit and torch.** The app launches with
`--server.fileWatcherType none`. Streamlit's file watcher walks
`torch.classes` and throws `Tried to instantiate class '__path__._path'`.
`demo.py` handles it; if you run `streamlit run app.py` directly, add the flag
yourself.

**First page load is slow.** Loading BERT or GPT-2 takes a few seconds. It's
cached for the rest of the session after that.

**Claude Desktop doesn't inherit your shell.** Both paths in
`claude_desktop_config.json` must be absolute, and the python you point at
must be the one where you ran `pip install mcp`. If you used a virtualenv,
point at the venv's python. This is the single most common setup failure —
check `~/Library/Logs/Claude/mcp*.log` (macOS) or
`%APPDATA%\Claude\logs\mcp*.log` (Windows) if it won't connect.

**Use your own API key.** There is no key in this repo and there shouldn't be
one in yours. Read it from the environment, and add `.env` and `_models/` to
`.gitignore` before your first commit.

---

## Questions

Open an issue, or reach out on LinkedIn. If you build one of the projects
above I'd like to see it — including the ones that didn't work.

"""
Talk mode — LLM demos for the CSE seminar.

Launch with:   python demo.py

Every model is read from ./_models, populated once by setup.py.
"""
import os
import pathlib

# These MUST be set before tiktoken or transformers are imported anywhere.
HERE = pathlib.Path(__file__).parent.resolve()
MODELS = HERE / "_models"
os.environ["TIKTOKEN_CACHE_DIR"] = str(MODELS / "tiktoken")
os.environ["HF_HOME"] = str(MODELS / "hf")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import math
import random
import streamlit as st

st.set_page_config(page_title="LLM demos", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------- style
st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 18px; }
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  h1 { font-size: 2.5rem !important; }
  h2 { font-size: 1.7rem !important; margin-top: 1.2rem !important; }
  .stSlider label, .stRadio label, .stTextInput label,
  .stSelectbox label, .stTextArea label { font-size: 1.05rem !important; }
  code, pre, .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

  .chip { display:inline-block; padding:6px 10px; margin:3px 2px;
          border-radius:3px; font-family:ui-monospace,Menlo,Consolas,monospace;
          font-size:1.15rem; white-space:pre; }
  .ids  { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.0rem;
          color:#5A6472; margin-top:8px; word-break:break-all; }

  table.big { border-collapse:collapse; width:100%; font-size:1.1rem; }
  table.big th { text-align:left; padding:10px 14px; border-bottom:2px solid #16202E;
                 font-weight:600; }
  table.big td { padding:9px 14px; border-bottom:1px solid #DDDAD3; }
  table.big td.num { font-family:ui-monospace,Menlo,Consolas,monospace;
                     text-align:right; font-weight:600; }

  .bar-row { display:grid; grid-template-columns:170px 1fr 92px;
             align-items:center; gap:14px; height:42px; }
  .bar-tok { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.25rem;
             text-align:right; }
  .bar-out { height:26px; background:#0B5C4E; }
  .bar-pct { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.15rem;
             font-weight:600; }
  .dim { color:#BFBCB5; }
  .dim .bar-out { background:none; box-shadow:inset 0 0 0 2px #CFCCC5; }
  .fired .bar-out { background:#C2410C; }
  .fired .bar-tok, .fired .bar-pct { color:#C2410C; }

  .verdict { padding:16px 20px; border-radius:3px; font-size:1.2rem;
             font-weight:600; margin-top:14px; }
  .good { background:#0B5C4E; color:#fff; }
  .bad  { background:#8A5A00; color:#fff; }

  .axis-wrap { margin:26px 0 10px; }
  .axis-line { position:relative; height:3px; background:#16202E; margin:0 6%; }
  .axis-end { position:absolute; top:-9px; width:3px; height:21px; background:#16202E; }
  .axis-labels { display:flex; justify-content:space-between; margin:0 4%;
                 font-size:1.0rem; font-weight:600; }
  .mk { position:absolute; transform:translateX(-50%); white-space:nowrap;
        text-align:center; }
  .mk .dot { width:15px; height:15px; border-radius:50%; margin:0 auto; }
  .mk .lbl { font-size:.92rem; margin-top:4px;
             font-family:ui-monospace,Menlo,monospace; }

  .gen-text { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.45rem;
              line-height:1.75; padding:20px 24px; background:#F1F0EC;
              border-radius:3px; min-height:90px; }
  .gen-text .new { background:#FFD9C7; padding:1px 2px; }
  .hidden-note { padding:24px; background:#F1F0EC; border-radius:3px;
                 font-size:1.2rem; text-align:center; color:#5A6472; }
</style>
""", unsafe_allow_html=True)

CHIP_COLORS = ["#E4EFEA", "#F2E7DC", "#E6EAF2", "#F0E9F2", "#EDF1DE"]
SERIES_COLORS = ["#0B5C4E", "#C2410C", "#2F4B8F", "#7A3E8F", "#8A6A1F",
                 "#1F7A6A", "#A33D2E", "#4A5568"]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner=False)
def load_encodings():
    import tiktoken
    return {n: tiktoken.get_encoding(n) for n in ["o200k_base", "cl100k_base"]}


@st.cache_resource(show_spinner=False)
def load_bert():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    return tok, model


@st.cache_resource(show_spinner=False)
def load_gpt2():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tok, model


# ---------------------------------------------------------------------- nav
st.sidebar.title("Talk mode")
PAGES = [
    "1 — Tokenization",
    "2 — Next token, live",
    "3 — Sampling levers",
    "4 — Prompt framing",
]
page = st.sidebar.radio("Demo", PAGES, label_visibility="collapsed")
st.sidebar.divider()
notes_on = st.sidebar.toggle("Presenter notes", value=True)
st.sidebar.caption(f"Models: `{MODELS.name}/`")
if not MODELS.exists():
    st.sidebar.error("`_models/` missing. Run `python setup.py` first.")


def notes(md):
    if notes_on:
        with st.expander("What to say", expanded=False):
            st.markdown(md)


# =============================================================== 1. TOKENIZE
if page == PAGES[0]:
    st.title("The model never sees your text")
    st.caption("Slides 10 and 11 · byte pair encoding")

    encs = load_encodings()
    presets = ["reverberere", "unbelievable", "48291", "Thiruvananthapuram"]
    cols = st.columns(len(presets) + 1)
    for i, val in enumerate(presets):
        if cols[i].button(val, use_container_width=True):
            st.session_state["tok_text"] = val

    text = st.text_input("Text to tokenize",
                         value=st.session_state.get("tok_text", "reverberere"),
                         key="tok_text")

    if text:
        for name, enc in encs.items():
            ids = enc.encode(text)
            pieces = [enc.decode([i]) for i in ids]
            st.subheader(name)
            chips = "".join(
                f'<span class="chip" style="background:{CHIP_COLORS[i % len(CHIP_COLORS)]}">'
                f'{esc(p) or "&#9251;"}</span>' for i, p in enumerate(pieces))
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown(f'<div class="ids">{ids}</div>', unsafe_allow_html=True)
            st.markdown(f"**{len(text)} characters &rarr; {len(ids)} tokens** &middot; "
                        f"{len(text)/max(1,len(ids)):.2f} chars per token")
            st.write("")

    st.divider()
    st.header("The token tax")

    PARAS = {
        "English": ("Chennai is the capital of Tamil Nadu. The city is known for its "
                    "long coastline, its classical music season, and a large software "
                    "industry. Millions of people pass through its central railway "
                    "station every year."),
        "Tamil": ("சென்னை தமிழ்நாட்டின் தலைநகரம். இந்த நகரம் அதன் நீண்ட கடற்கரை, "
                  "பாரம்பரிய இசை விழா காலம், மற்றும் பெரிய மென்பொருள் தொழில் "
                  "ஆகியவற்றுக்கு அறியப்படுகிறது. ஒவ்வொரு ஆண்டும் லட்சக்கணக்கான "
                  "மக்கள் அதன் மத்திய இரயில் நிலையம் வழியாக பயணிக்கிறார்கள்."),
        "Hindi": ("तमिलनाडु की राजधानी चेन्नई है। यह शहर अपने लंबे समुद्र तट, अपने "
                  "शास्त्रीय संगीत के मौसम, और एक बड़े सॉफ़्टवेयर उद्योग के लिए जाना "
                  "जाता है। हर साल लाखों लोग इसके मुख्य रेलवे स्टेशन से गुजरते हैं।"),
    }
    which = st.selectbox("Tokenizer", list(encs.keys()))
    enc = encs[which]
    html = ["<table class='big'><tr><th>Language</th><th>Characters</th>"
            "<th>Tokens</th><th>Chars / token</th><th>vs English</th></tr>"]
    base = None
    for lang, para in PARAS.items():
        n = len(enc.encode(para))
        base = base or n
        html.append(f"<tr><td>{lang}</td><td class='num'>{len(para)}</td>"
                    f"<td class='num'>{n}</td><td class='num'>{len(para)/n:.2f}</td>"
                    f"<td class='num'>{n/base:.1f}&times;</td></tr>")
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    notes("""
Point at the chips, not the numbers. `reverberere` arrived as a handful of
coloured blocks — the model was never given eleven letters, so counting
characters means reconstructing something it never had.

Then `48291`. Watch where the digits split. That irregularity is why
multi-digit arithmetic is unreliable.

Switch the tokenizer dropdown and the split changes. **The boundary is a
property of the dictionary, not of the word** — that's the slide 11 point.

Take a word from the room. This is the page where the audience drives.

**Deck fix:** slide 10 says "not 12 individual letters". It's 11.
""")

# =============================================================== 2. GENERATOR
elif page == PAGES[1]:
    st.title("Be the model")
    st.caption("Slides 16 and 17 · a real model, one token at a time")

    st.session_state.setdefault("gen_ids", None)
    st.session_state.setdefault("gen_new", 0)

    prompt = st.text_input("Prompt", "Chennai is a city famous for its")
    c1, c2, c3 = st.columns([1, 1, 2])
    start = c1.button("Start / reset", type="primary", use_container_width=True)
    hide = c2.toggle("Hide candidates", value=False,
                     help="Ask the room to guess, then switch this off")
    temp = c3.slider("Temperature", 0.1, 2.0, 1.0, 0.05)

    with st.spinner("Loading gpt2…"):
        tok, model = load_gpt2()
    import torch
    import torch.nn.functional as F

    if start or st.session_state.gen_ids is None:
        st.session_state.gen_ids = tok(prompt)["input_ids"]
        st.session_state.gen_new = 0

    @st.cache_data(show_spinner=False)
    def candidates(ids_tuple, temperature, k=8):
        ids = torch.tensor([list(ids_tuple)])
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        top = torch.topk(probs, k)
        return [(int(i), tok.decode([int(i)]), float(p))
                for p, i in zip(top.values, top.indices)]

    ids = st.session_state.gen_ids
    cands = candidates(tuple(ids), temp)

    n_new = st.session_state.gen_new
    if n_new:
        old = tok.decode(ids[:len(ids) - n_new])
        new = tok.decode(ids[len(ids) - n_new:])
        body = f"{esc(old)}<span class='new'>{esc(new)}</span>"
    else:
        body = esc(tok.decode(ids))
    st.markdown(f"<div class='gen-text'>{body}</div>", unsafe_allow_html=True)
    st.caption(f"{len(ids)} tokens total &middot; {n_new} generated here")

    st.markdown("### What comes next")
    if hide:
        st.markdown("<div class='hidden-note'>Take three guesses from the room, "
                    "then switch off <b>Hide candidates</b>.</div>",
                    unsafe_allow_html=True)
    else:
        peak = max(p for _, _, p in cands) or 1.0
        for tid, text, p in cands:
            b1, b2, b3 = st.columns([2, 5, 1])
            shown = text.replace(" ", "\u2423") if text != text.strip() else text
            if b1.button(shown or "\u2423", key=f"pick{tid}",
                         use_container_width=True):
                st.session_state.gen_ids = ids + [tid]
                st.session_state.gen_new += 1
                st.rerun()
            b2.markdown(f"<div style='height:26px;background:#0B5C4E;"
                        f"width:{p/peak*100:.1f}%'></div>", unsafe_allow_html=True)
            b3.markdown(f"<div class='bar-pct'>{p*100:.1f}%</div>",
                        unsafe_allow_html=True)

    st.divider()
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Take the top token", use_container_width=True):
        st.session_state.gen_ids = ids + [cands[0][0]]
        st.session_state.gen_new += 1
        st.rerun()
    if a2.button("Sample one", use_container_width=True):
        pick = random.choices([c[0] for c in cands],
                              weights=[c[2] for c in cands])[0]
        st.session_state.gen_ids = ids + [pick]
        st.session_state.gen_new += 1
        st.rerun()
    if a3.button("Run 10 more", use_container_width=True):
        cur = list(ids)
        for _ in range(10):
            cs = candidates(tuple(cur), temp)
            cur.append(random.choices([c[0] for c in cs],
                                      weights=[c[2] for c in cs])[0])
        st.session_state.gen_ids = cur
        st.session_state.gen_new += 10
        st.rerun()
    if a4.button("Undo", use_container_width=True, disabled=n_new == 0):
        st.session_state.gen_ids = ids[:-1]
        st.session_state.gen_new = max(0, n_new - 1)
        st.rerun()

    notes("""
This replaces the Transformer Explainer website, so slides 16 and 17 have no
network dependency — and the prompt is yours, which matters more.

**Run it like a game.**

1. Switch on **Hide candidates**. Read the prompt aloud. Take three guesses
   from the room and write them on the board.
2. Switch it off. Their guesses are usually in the top eight. That is the
   realisation: they and the model are doing the same thing.
3. Now click tokens *yourself*, several times, and watch the sentence build.
   You are the sampler. Say that out loud.
4. Then hit **Take the top token** repeatedly. GPT-2 falls into a loop and
   repeats itself — which is exactly why nobody ships greedy decoding for
   prose, and it hands you the temperature conversation for free.
5. Crank temperature to 2.0 and **Run 10 more**. It falls apart. Drop to 0.3
   and run again. Same model, same prompt.

Take a prompt from the room at some point. "Chennai traffic is" works well.

Be upfront that this is GPT-2 from 2019 and it is small, so the prose will be
mediocre. That's a feature here — the *mechanism* is identical in a frontier
model, and watching it done badly makes the mechanism more visible, not less.
""")

# =============================================================== 3. SAMPLING
elif page == PAGES[2]:
    st.title("Temperature, top-k, top-p")
    st.caption("Slide 19 · pure arithmetic, no model needed")

    VOCAB = [("beaches", .34), ("temples", .27), ("food", .15), ("music", .09),
             ("heat", .06), ("filter coffee", .04), ("traffic", .02),
             ("marina", .015), ("culture", .01), ("rain", .005)]
    NAMES = [v[0] for v in VOCAB]
    LOGITS = [math.log(v[1]) for v in VOCAB]

    st.markdown(r"### Chennai is a city famous for its \_\_\_\_")

    st.session_state.setdefault("draws", [])
    st.session_state.setdefault("fired", -1)
    st.session_state.setdefault("samp_T", 1.0)
    st.session_state.setdefault("samp_K", 10)
    st.session_state.setdefault("samp_P", 1.0)

    def apply_preset(t, k, p):
        st.session_state.samp_T = t
        st.session_state.samp_K = k
        st.session_state.samp_P = p
        st.session_state.fired = -1

    PRESETS = [("Greedy — code and JSON", 0.01, 10, 1.0),
               ("Balanced", 0.7, 10, 0.9),
               ("Creative", 1.4, 10, 1.0),
               ("Too hot", 2.0, 10, 1.0)]
    for col, (label, t, k, p) in zip(st.columns(len(PRESETS)), PRESETS):
        col.button(label, on_click=apply_preset, args=(t, k, p),
                   use_container_width=True)

    c1, c2, c3 = st.columns(3)
    T = c1.slider("Temperature", 0.01, 2.0, step=0.01, key="samp_T")
    K = c2.slider("Top-k", 1, 10, step=1, key="samp_K")
    P = c3.slider("Top-p", 0.05, 1.0, step=0.01, key="samp_P")

    def distribution(T, k, p):
        scaled = [l / T for l in LOGITS]
        m = max(scaled)
        ex = [math.exp(v - m) for v in scaled]
        s = sum(ex)
        probs = [v / s for v in ex]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        keep = set(order[:k])
        pool, cum = [], 0.0
        for i in order:
            if i not in keep:
                continue
            pool.append(i)
            cum += probs[i]
            if cum >= p:
                break
        mass = sum(probs[i] for i in pool)
        return [probs[i] / mass if i in pool else 0.0
                for i in range(len(probs))], set(pool)

    final, pool = distribution(T, K, P)
    peak = max(final) or 1.0
    entropy = -sum(v * math.log2(v) for v in final if v > 0)

    b1, b2, b3 = st.columns([1, 1, 2])
    if b1.button("Sample one token", type="primary", use_container_width=True):
        pick = random.choices(range(len(NAMES)), weights=final)[0]
        st.session_state.fired = pick
        st.session_state.draws.append(pick)
    if b2.button("Sample 20", use_container_width=True):
        st.session_state.fired = -1
        st.session_state.draws += random.choices(range(len(NAMES)),
                                                 weights=final, k=20)
    if b3.button("Clear draws", use_container_width=True):
        st.session_state.draws = []
        st.session_state.fired = -1

    rows = []
    for i, name in enumerate(NAMES):
        inpool = i in pool
        cls = "fired" if i == st.session_state.fired else ("" if inpool else "dim")
        if inpool and final[i] >= 0.005:
            w, pct = final[i] / peak * 100, f"{final[i]*100:.1f}%"
        elif inpool:
            w, pct = 1.5, "~0%"
        else:
            w, pct = 4, "&mdash;"
        rows.append(f"<div class='bar-row {cls}'><div class='bar-tok'>{name}</div>"
                    f"<div><div class='bar-out' style='width:{w:.1f}%'></div></div>"
                    f"<div class='bar-pct'>{pct}</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Tokens the model may pick", f"{len(pool)} of {len(NAMES)}")
    m2.metric("Entropy — how undecided it is", f"{entropy:.2f} bits")
    if st.session_state.draws:
        recent = st.session_state.draws[-20:]
        m3.metric(f"Distinct in last {len(recent)} draws", f"{len(set(recent))}")
        st.markdown("**Draws:** " + "  &middot;  ".join(NAMES[i] for i in recent))

    notes("""
Run it in this order:

1. **Greedy** preset, then *Sample 20*. Distinct count is 1. That's your
   slide 17 determinism claim, demonstrated instead of asserted.
2. **Creative**, *Sample 20*. Five or six distinct tokens.
3. **Too hot**. "rain" and "marina" get picked — broken output from the inside.
4. Set **top-k = 3** and drag temperature across its range. The pool stays at
   3 whatever shape the distribution takes. A hard ceiling.
5. Set **top-k = 10**, **top-p = 0.9**, then drag temperature 0.6 → 1.6. The
   pool goes from 3 tokens to 7 **on its own**. Same `p`, different pool.

**Step 5 is the demonstration.** Sweeping top-p at a fixed temperature also
moves the boundary, but it doesn't show *why* top-p exists. Holding `p` steady
and changing the shape underneath it does.

This is the fixed-distribution version of page 3. If you only have time for
one, use page 3 — same idea, real model.
""")

# ============================================================== 4. PROMPTING
else:
    st.title("The prompt shapes the distribution")
    st.caption("Slide 18 · same question, different framing")

    st.markdown("""
Prompting does not change the weights. It changes which tokens are likely next.
This measures that directly: how much probability mass lands on the **answers
you actually want**, versus everything else.
""")

    labels_raw = st.text_input("Answers you want (comma separated)",
                               "positive, negative")
    question = st.text_input("The input to classify",
                             "The food was cold and the service was slow.")

    c1, c2 = st.columns(2)
    zero = c1.text_area("Zero-shot prompt", height=240,
                        value="Review: {input}\nSentiment:")
    few = c2.text_area(
        "Few-shot prompt", height=240,
        value="Review: Best dosa in the city.\nSentiment: positive\n"
              "Review: Waited an hour and it was burnt.\nSentiment: negative\n"
              "Review: Lovely staff, great coffee.\nSentiment: positive\n"
              "Review: {input}\nSentiment:")
    st.caption("`{input}` is replaced by the text above.")

    if st.button("Measure both", type="primary"):
        with st.spinner("Loading gpt2…"):
            tok, model = load_gpt2()
        import torch
        import torch.nn.functional as F

        labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

        def analyse(template):
            text = template.replace("{input}", question)
            ids = tok(text, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                logits = model(ids).logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            top = torch.topk(probs, 8)
            top_list = [(tok.decode([int(i)]), float(p))
                        for p, i in zip(top.values, top.indices)]
            mass, per = 0.0, []
            for lab in labels:
                tid = tok(" " + lab)["input_ids"][0]
                p = float(probs[tid])
                per.append((lab, p))
                mass += p
            return top_list, mass, per

        if not labels:
            st.error("Give at least one answer you want.")
        else:
            try:
                res = {"Zero-shot": analyse(zero), "Few-shot": analyse(few)}
            except Exception as e:
                st.error(f"Could not run that: {e}")
            else:
                for col, (name, (top_list, mass, per)) in zip(st.columns(2),
                                                              res.items()):
                    with col:
                        st.subheader(name)
                        st.metric("Mass on the answers you wanted",
                                  f"{mass*100:.1f}%")
                        for lab, p in per:
                            st.markdown(f"&nbsp;&nbsp;`{lab}` &mdash; "
                                        f"**{p*100:.2f}%**")
                        st.markdown("**Top 8 next tokens**")
                        peak = max(p for _, p in top_list) or 1.0
                        bars = []
                        for text, p in top_list:
                            hit = any(text.strip().lower() == l.lower()
                                      for l in labels)
                            colr = "#C2410C" if hit else "#0B5C4E"
                            bars.append(
                                f"<div class='bar-row'><div class='bar-tok' "
                                f"style='font-size:1.0rem'>{esc(repr(text))}</div>"
                                f"<div><div class='bar-out' "
                                f"style='width:{p/peak*100:.1f}%;"
                                f"background:{colr}'></div></div>"
                                f"<div class='bar-pct' style='font-size:1.0rem'>"
                                f"{p*100:.1f}%</div></div>")
                        st.markdown("".join(bars), unsafe_allow_html=True)

                z, f = res["Zero-shot"][1], res["Few-shot"][1]
                if z > 1e-9:
                    st.markdown(
                        f"<div class='verdict {'good' if f > z else 'bad'}'>"
                        f"Examples moved the mass on valid answers from "
                        f"{z*100:.1f}% to {f*100:.1f}% &mdash; a {f/z:.1f}&times; "
                        f"shift.</div>", unsafe_allow_html=True)

    notes("""
The headline number is **mass on the answers you wanted**. Zero-shot, GPT-2
spreads its probability over newlines, punctuation and random words. Add three
examples and the mass concentrates on `positive` / `negative`.

Say the engineering point plainly: **few-shot prompting did not make the model
smarter, it made the output shape predictable.** That is why it matters in
production — you are buying parseability, not intelligence. Same reason
temperature 0 matters for JSON.

Things to try live:
- Delete the examples one at a time and watch the mass fall.
- Change the wanted answers to `good, bad` and re-run with the same examples.
  The mass collapses, because the examples taught a *vocabulary*, not a concept.
  That's the sharpest moment on this page.
- Feed it a review in Tamil and watch it fail, which loops back to page 1.

Caveat to state out loud: GPT-2 is a 2019 base model with no instruction
tuning, so role prompts and constraint prompts do almost nothing here. That's
why this page compares zero-shot against few-shot rather than all four
strategies on your slide. With an instruction-tuned model the other two work,
but the effect is much harder to show as a single number.
""")

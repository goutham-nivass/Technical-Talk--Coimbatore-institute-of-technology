# Talk mode — LLM demos

One UI for all four demos in the first half of the seminar. Pick a demo in
the sidebar, hit a button, the code runs behind it.

| Sidebar | Slide | What it shows |
|---|---|---|
| 1 — Tokenization | 10, 11 | Token chips, integer IDs, the English/Tamil/Hindi token tax |
| 2 — Contextual embeddings | 13, 14 | `bank` traced through three stages, with controls |
| 3 — Winograd, measured | 12 | A model resolving trophy vs suitcase, as probabilities |
| 4 — Sampling levers | 19 | Temperature, top-k, top-p reshaping the candidate pool |

Every model lives in **`_models/`** next to the app. Nothing is downloaded at
runtime.

---

## First time, at home, on wifi

```bash
pip install -r requirements.txt
python setup.py
```

`setup.py` pulls two tokenizer vocabularies and two models into `_models/`
and prints the folder size. Expect a few hundred MB.

Then confirm it works without a network — **turn your wifi off** and run:

```bash
python setup.py --verify
```

This forces every library into offline mode, so a silent re-download can't
hide a missing file. If it prints `PASSED`, you're set.

## On the day

```bash
python demo.py
```

That's the only command. It points the cache at `_models/`, sets the flags
that keep Streamlit stable, and opens your browser.

---

## Things worth knowing

**First click on pages 2 and 3 is slow.** Loading BERT or GPT-2 takes a few
seconds. After that it's cached for the whole session, so switching between
pages is instant. Click through both pages once before the audience arrives
and the delay never happens on stage.

**torch isn't imported until you need it.** Page 1 and page 4 never touch it,
so the app starts fast and the tokenizer demo works even if the torch install
went wrong.

**`--server.fileWatcherType none` is set deliberately.** Streamlit's file
watcher walks `torch.classes` and throws `Tried to instantiate class
'__path__._path'`. Turning the watcher off is the standard fix and costs
nothing, since you won't be editing code mid-talk.

**Presenter notes are a sidebar toggle.** Each page has a collapsed "What to
say" panel with the wording and the order to run things in. Turn the toggle
off before you project if you'd rather not have it visible.

**Page 3 may not come out clean.** GPT-2 is small and Winograd schemas are
genuinely hard for it, so the diagonal can fail. The app tells you which way
it went rather than hiding it, and offers a line to use if it fails. Run it
once before the talk so you know which version you're presenting.

---

## Presenting order for page 4

The sampling page is the one with a specific script, because the top-k versus
top-p contrast is easy to get wrong:

1. **Greedy** preset, then *Sample 20*. Distinct count is 1. That's your
   slide 17 determinism claim, demonstrated.
2. **Creative**, *Sample 20*. Five or six distinct.
3. **Too hot**. `rain` and `marina` start getting picked — broken output from
   the inside.
4. Set **top-k = 3** and drag temperature across its range. Pool stays at 3
   whatever the shape. A hard ceiling.
5. Set **top-k = 10**, **top-p = 0.9**, then drag temperature 0.6 → 1.6. Pool
   goes from 3 tokens to 7 **on its own**. Same `p`, different pool.

Step 5 is the demonstration. Sweeping top-p at a fixed temperature also moves
the boundary, but it doesn't show *why* top-p exists — holding `p` steady and
changing the shape underneath it does.

---

## Deck fixes this surfaced

- Slide 10 says "not 12 individual letters". `reverberere` has **11**.
- Slide 10's "4x–8x" token tax — replace with the measured numbers from page 1.
- Nine slides render raw LaTeX (`$N \times M$`). Find-and-replace those.

---

## Checklist

- [ ] `python setup.py --verify` passed with wifi off
- [ ] Pages 2 and 3 clicked once so models are warm
- [ ] Page 3 result known — clean, or line prepared
- [ ] Real token counts written onto slide 10
- [ ] Browser zoom at 125–150% for the projector
- [ ] Presenter notes toggle set how you want it
- [ ] Screen recording of all four pages, as fallback

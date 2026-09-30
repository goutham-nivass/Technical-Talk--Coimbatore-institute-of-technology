"""
Run this ONCE, at home, on good wifi.

Downloads every model and vocabulary file the demo app needs into a single
folder next to this script:

    ./_models/

On talk day the app points at that folder and never reaches the network.

    python setup.py            # download everything
    python setup.py --verify   # then turn wifi OFF and run this
"""
import os
import sys
import pathlib

HERE = pathlib.Path(__file__).parent.resolve()
MODELS = HERE / "_models"

# Must be set before tiktoken / transformers are imported.
os.environ["TIKTOKEN_CACHE_DIR"] = str(MODELS / "tiktoken")
os.environ["HF_HOME"] = str(MODELS / "hf")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
(MODELS / "tiktoken").mkdir(parents=True, exist_ok=True)
(MODELS / "hf").mkdir(parents=True, exist_ok=True)

VERIFY = "--verify" in sys.argv
if VERIFY:
    # Force offline so a silent re-download cannot hide a missing cache entry.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

ENCODINGS = ["o200k_base", "cl100k_base"]
HF_MODELS = [("bert-base-uncased", "encoder"), ("gpt2", "causal")]


def main():
    tag = "[verify] " if VERIFY else ""
    failures = []

    print(f"\n{tag}Tokenizer vocabularies")
    try:
        import tiktoken
        for name in ENCODINGS:
            enc = tiktoken.get_encoding(name)
            print(f"  {name:<14} ok    {enc.n_vocab:,} tokens")
    except Exception as e:
        failures.append(f"tiktoken: {e}")
        print(f"  FAILED  {e}")

    print(f"\n{tag}Models")
    try:
        from transformers import (AutoTokenizer, AutoModel,
                                  AutoModelForCausalLM)
        for name, kind in HF_MODELS:
            AutoTokenizer.from_pretrained(name)
            if kind == "causal":
                AutoModelForCausalLM.from_pretrained(name)
            else:
                AutoModel.from_pretrained(name)
            print(f"  {name:<20} ok")
    except Exception as e:
        failures.append(f"transformers: {e}")
        print(f"  FAILED  {e}")

    total = sum(f.stat().st_size for f in MODELS.rglob("*") if f.is_file())
    print(f"\nFolder    {MODELS}")
    print(f"Size      {total / 1e6:.0f} MB")

    if failures:
        print("\nNOT READY:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    if VERIFY:
        print("\nOffline check PASSED. The app will run with no network.")
    else:
        print("\nDone. Now turn wifi off and run:  python setup.py --verify")


if __name__ == "__main__":
    main()

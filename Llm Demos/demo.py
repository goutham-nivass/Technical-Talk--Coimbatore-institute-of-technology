"""
Talk mode launcher.

    python demo.py

Checks that the model folder exists, then starts the Streamlit app with the
flags that keep it stable during a live talk.
"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent.resolve()
MODELS = HERE / "_models"
APP = HERE / "app.py"


def main():
    if not APP.exists():
        sys.exit(f"app.py not found next to {__file__}")

    if not MODELS.exists():
        print("\n  _models/ not found.\n")
        print("  Run this first, on good wifi:\n")
        print("      python setup.py\n")
        print("  Then verify it works with wifi off:\n")
        print("      python setup.py --verify\n")
        if input("  Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit(0)

    env = dict(os.environ)
    env["TIKTOKEN_CACHE_DIR"] = str(MODELS / "tiktoken")
    env["HF_HOME"] = str(MODELS / "hf")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        # Streamlit's file watcher walks torch.classes and throws
        # "Tried to instantiate class '__path__._path'". Turning it off is the
        # standard fix and costs nothing here, since you are not editing code
        # during the talk.
        "--server.fileWatcherType", "none",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
    ]

    print("\n  Starting talk mode. Browser opens automatically.")
    print("  Stop with Ctrl-C.\n")
    try:
        subprocess.run(cmd, env=env, cwd=str(HERE))
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()

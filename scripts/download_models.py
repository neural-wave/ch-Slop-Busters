"""Download the pinned Hugging Face models listed in src/unified_harness/models.toml.

Development-time only: inference never downloads anything. Each model lands in <dest>/<dir> together
with a `.revision` marker that `unified-harness doctor` compares with the pin. Ollama models are pulled
separately (`ollama pull <name>`).

    uv run --extra download python scripts/download_models.py            # into ./models
    uv run --extra download python scripts/download_models.py --dest /srv/models --only gliner_pii
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from unified_harness.environment import REVISION_FILE, check_hf_model, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="models", help="models directory (default: ./models)")
    parser.add_argument("--only", action="append", help="manifest key to download (repeatable)")
    args = parser.parse_args()

    dest = Path(args.dest)
    entries = load_manifest()["huggingface"]
    keys = args.only or list(entries)
    unknown = sorted(set(keys) - set(entries))
    if unknown:
        parser.error(f"unknown model key(s) {unknown}; known: {sorted(entries)}")

    for key in keys:
        entry = entries[key]
        if check_hf_model(entry, dest).ok:
            print(f"[skip] {entry['repo']} already at {entry['revision'][:12]}")
            continue
        folder = dest / entry["dir"]
        folder.mkdir(parents=True, exist_ok=True)
        for name in entry["files"]:
            hf_hub_download(entry["repo"], name, revision=entry["revision"], local_dir=folder)
        (folder / REVISION_FILE).write_text(entry["revision"] + "\n", encoding="utf-8")
        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        print(f"[done] {entry['repo']} @ {entry['revision'][:12]} -> {folder} ({size / 2**20:,.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

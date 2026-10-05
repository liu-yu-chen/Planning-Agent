"""Build the retrieval index, apply citation-age eligibility, then build the graph."""

from __future__ import annotations

import argparse
from datetime import datetime
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-docs", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--restart-index", action="store_true", help="Archive the current partial index before rebuilding from the deduplicated corpus")
    args = parser.parse_args()
    index_dir = HERE / "index"
    if args.restart_index and index_dir.exists():
        archived = HERE / f"index_pre_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(index_dir), str(archived))
        print(f"Archived previous partial index to {archived}", flush=True)
    subprocess.run([sys.executable, str(HERE / "deduplicate_corpus.py")], check=True, cwd=HERE.parents[1])
    subprocess.run([
        sys.executable, str(HERE / "build_indexes.py"),
        "--batch-docs", str(args.batch_docs), "--device", args.device,
    ], check=True, cwd=HERE.parents[1])
    subprocess.run([sys.executable, str(HERE / "apply_citation_filter.py")], check=True, cwd=HERE.parents[1])
    subprocess.run([sys.executable, str(HERE / "build_graph.py")], check=True, cwd=HERE.parents[1])
    print("Retrieval index, citation eligibility filter, and OpenAlex graph are ready. Starting paired RAG benchmark.", flush=True)
    subprocess.run([sys.executable, str(HERE / "benchmark.py")], check=True, cwd=HERE.parents[1])
    print("Indexing and paired RAG benchmark are complete.", flush=True)


if __name__ == "__main__":
    main()

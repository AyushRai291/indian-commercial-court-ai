#!/usr/bin/env python3
"""Download and validate an official-court judgment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.acquisition import JudgmentDownloader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate official judgment PDFs with resumable progress."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/judgments_pilot.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/judgments"),
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("data/failed/judgment_downloads.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/manifests/judgments_pilot_audit.json"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--min-text-characters", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    headers = {
        "User-Agent": "IndianCommercialCourtCorpus/0.1 (research; respectful downloader)"
    }
    with httpx.Client(
        timeout=httpx.Timeout(args.timeout),
        follow_redirects=True,
        headers=headers,
    ) as client:
        downloader = JudgmentDownloader(
            client,
            retries=args.retries,
            max_bytes=args.max_bytes,
            min_text_characters=args.min_text_characters,
        )
        audit = downloader.run(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            failure_log=args.failure_log,
            audit_path=args.audit,
        )

    counts = audit.to_dict(manifest_path=args.manifest)["counts"]
    print(json.dumps(counts, sort_keys=True))
    failures = sum(count for name, count in counts.items() if name != "downloaded")
    return 0 if counts["downloaded"] and failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

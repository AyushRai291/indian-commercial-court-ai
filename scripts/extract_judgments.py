#!/usr/bin/env python3
"""Convert the validated pilot PDFs to canonical ingestion JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from legal_rag.extraction import PilotCorpusExtractor  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract validated pilot PDFs to canonical JSONL."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/judgments_pilot.jsonl"),
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/judgments")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/judgments_pilot.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoints/judgments_pilot_extraction.json"),
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=Path("data/failed/judgment_extraction.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/manifests/judgments_pilot_extraction_audit.json"),
    )
    parser.add_argument("--min-text-characters", type=int, default=200)
    parser.add_argument("--restart", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = PilotCorpusExtractor(
            min_text_characters=args.min_text_characters
        ).run(
            manifest_path=args.manifest,
            raw_dir=args.raw_dir,
            output_path=args.output,
            checkpoint_path=args.checkpoint,
            failure_log=args.failure_log,
            audit_path=args.audit,
            restart=args.restart,
        )
    except Exception as error:
        print(f"Extraction could not start: {error}", file=sys.stderr)
        return 1

    counts = audit.to_dict(
        manifest_path=args.manifest, output_path=args.output
    )["counts"]
    print(json.dumps(counts, sort_keys=True))
    return 0 if counts["extracted"] and not (
        counts["failures"] or counts["missing_metadata"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export prompt versions from a bookwiki database into markdown files."""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PromptRow = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export prompt summaries and templates from a bookwiki SQLite database "
            "into markdown files grouped by prompt key."
        )
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("prompts"),
        help="Directory to write exported prompt files (default: ./prompts).",
    )
    return parser.parse_args()


def load_prompts(connection: sqlite3.Connection) -> Dict[str, List[PromptRow]]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT key, create_time, summary, template
        FROM prompt
        ORDER BY key ASC, create_time ASC
        """
    )
    grouped: Dict[str, List[PromptRow]] = defaultdict(list)
    for row in cursor.fetchall():
        key, create_time, summary, template = row
        grouped[key].append((key, summary, template))
    return grouped


def write_prompt_versions(
    output_dir: Path, key: str, versions: Iterable[PromptRow]
) -> None:
    key_dir = output_dir / key
    key_dir.mkdir(parents=True, exist_ok=True)

    for index, (_, summary, template) in enumerate(versions):
        filename = f"{index:02d}.md"
        content = "# Summary\n\n{summary}\n\n# Template\n\n{template}\n".format(
            summary=summary, template=template
        )
        (key_dir / filename).write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.db_path.is_file():
        raise SystemExit(f"Database file not found: {args.db_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.db_path) as connection:
        grouped_prompts = load_prompts(connection)

    for key in sorted(grouped_prompts):
        write_prompt_versions(args.output_dir, key, grouped_prompts[key])


if __name__ == "__main__":
    main()

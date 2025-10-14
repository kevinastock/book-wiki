#!/usr/bin/env python3
"""Print prompt keys and the latest summary for each key."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

SummaryRow = Tuple[str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List prompt keys from a bookwiki SQLite database along with the "
            "summary from the latest version of each key."
        )
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the SQLite database file.",
    )
    return parser.parse_args()


def fetch_latest_summaries(connection: sqlite3.Connection) -> Iterable[SummaryRow]:
    cursor = connection.cursor()
    cursor.execute(
        """
        WITH prompt_bounds AS (
            SELECT key,
                   MIN(create_time) AS first_time,
                   MAX(create_time) AS latest_time
            FROM prompt
            GROUP BY key
        )
        SELECT pb.key, pb.first_time, p.summary
        FROM prompt_bounds pb
        JOIN prompt p
          ON p.key = pb.key AND p.create_time = pb.latest_time
        ORDER BY pb.first_time ASC
        """
    )
    yield from cursor.fetchall()


def main() -> None:
    args = parse_args()
    if not args.db_path.is_file():
        raise SystemExit(f"Database file not found: {args.db_path}")

    with sqlite3.connect(args.db_path) as connection:
        rows = list(fetch_latest_summaries(connection))

    for key, _, summary in rows:
        print(f"## {key}\n\n*{summary}*\n\n")


if __name__ == "__main__":
    main()

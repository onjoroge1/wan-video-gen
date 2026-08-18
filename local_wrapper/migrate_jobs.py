#!/usr/bin/env python3
"""Import legacy outputs/jobs.json state into the configured Postgres database."""

import argparse
import os
import shlex
from pathlib import Path

from persistence import JsonJobRepository, PostgresJobRepository


def configured_database_url():
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "DATABASE_URL":
            values = shlex.split(value, comments=True, posix=True)
            return values[0] if values else None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parent / "outputs" / "jobs.json",
    )
    parser.add_argument("--database-url", default=configured_database_url())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.database_url:
        parser.error("set DATABASE_URL or pass --database-url")
    if not args.source.is_file():
        parser.error(f"source file does not exist: {args.source}")
    jobs = JsonJobRepository(args.source).load_all()
    print(f"validated {len(jobs)} job(s) from {args.source}")
    if args.dry_run:
        return 0

    destination = PostgresJobRepository(args.database_url)
    destination.initialize()
    destination.save_all(jobs)
    destination.close()
    print(f"imported {len(jobs)} job(s) into Postgres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

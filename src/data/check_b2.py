"""Live object-storage checks for the data pipeline.

# Upload and read back a small test object under test/noe/ (handoff check, #82)
python -m src.data.check_b2 roundtrip

# Download the published dataset and verify it against its manifest (#85)
python -m src.data.check_b2 verify-dataset --prefix processed/movielens-1m/v1/
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime

from src.data.prepare_movielens import verify_published
from src.data.storage import (
    PROCESSED_PREFIX,
    TEST_PREFIX,
    StorageSettings,
    make_client,
    normalize_prefix,
    upload_verified,
)


def roundtrip(client, bucket: str, prefix: str = TEST_PREFIX, now: datetime | None = None):
    """Write a uniquely named test object, read it back, and confirm it is listed."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    key = f"{normalize_prefix(prefix)}roundtrip-{stamp}.txt"
    result = upload_verified(client, bucket, key, f"roundtrip {stamp}\n".encode(), "text/plain")
    listed = client.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", [])
    if not any(item["Key"] == key for item in listed):
        raise RuntimeError(f"{key} was uploaded but is not listed.")
    return {**result, "bucket": bucket, "listed": True}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", help="Defaults to B2_BUCKET.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("roundtrip").add_argument("--prefix", default=TEST_PREFIX)
    commands.add_parser("verify-dataset").add_argument("--prefix", default=PROCESSED_PREFIX)
    args = parser.parse_args(argv)

    settings = StorageSettings.from_env(bucket=args.bucket)
    client = make_client(settings)
    if args.command == "roundtrip":
        result = roundtrip(client, settings.bucket, args.prefix)
    else:
        result = verify_published(client, settings.bucket, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Upload the original MovieLens 1M ``.dat`` files to object storage and verify them.

Raw objects are immutable: an existing object is never overwritten. If it already
exists it is downloaded and checked against the local file instead.

    python -m src.data.upload_raw --source-dir data/raw/ml-1m
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from src.data.storage import (
    RAW_PREFIX,
    StorageSettings,
    head,
    make_client,
    normalize_prefix,
    sha256_hex,
    upload_verified,
    verify_object,
)

RAW_FILES = ("users.dat", "movies.dat", "ratings.dat")


def upload_raw_files(client, bucket: str, source_dir: Path, prefix: str = RAW_PREFIX) -> list:
    prefix = normalize_prefix(prefix)
    missing = [name for name in RAW_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files in {source_dir}: {', '.join(missing)}")

    results = []
    for name in RAW_FILES:
        data = (source_dir / name).read_bytes()
        key = f"{prefix}{name}"
        if head(client, bucket, key) is None:
            result = upload_verified(client, bucket, key, data)
            result["status"] = "uploaded"
        else:
            result = verify_object(client, bucket, key, sha256_hex(data), len(data))
            result["status"] = "already present, verified"
        results.append(result)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--prefix", default=RAW_PREFIX)
    parser.add_argument("--bucket", help="Defaults to B2_BUCKET.")
    args = parser.parse_args(argv)

    settings = StorageSettings.from_env(bucket=args.bucket)
    results = upload_raw_files(make_client(settings), settings.bucket, args.source_dir, args.prefix)
    print(json.dumps({"bucket": settings.bucket, "objects": results}, indent=2))


if __name__ == "__main__":
    main()

"""Convert the raw MovieLens 1M ``.dat`` files into a validated, versioned CSV dataset.

Reads ``users.dat``, ``movies.dat`` and ``ratings.dat`` (``::`` delimited, Latin-1)
from object storage (or a local directory for development), validates them, joins
ratings with users and movies, and writes ``ratings.csv``, ``users.csv``,
``movies.csv`` and ``manifest.json``. With ``--publish`` the files are uploaded to
the versioned processed prefix (manifest last) and verified by downloading them again.

    python -m src.data.prepare_movielens --output-dir data/processed/movielens-1m/v1
    python -m src.data.prepare_movielens --output-dir data/processed/movielens-1m/v1 --publish
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.data.storage import (
    PROCESSED_PREFIX,
    RAW_PREFIX,
    StorageSettings,
    download,
    head,
    make_client,
    normalize_prefix,
    sha256_hex,
    upload_verified,
    verify_object,
)

DATASET = "movielens-1m"
SOURCE_ENCODING = "latin-1"
DELIMITER = "::"
USER_COLUMNS = ["user_id", "gender", "age_code", "occupation_code", "zip_code"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
OUTPUT_COLUMNS = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp",
    "title",
    "genres",
    "gender",
    "age_code",
    "occupation_code",
    "zip_code",
]
SOURCE_FILES = {"users": "users.dat", "movies": "movies.dat", "ratings": "ratings.dat"}
OUTPUT_FILES = ("ratings.csv", "users.csv", "movies.csv")
MANIFEST_NAME = "manifest.json"
STRING_COLUMNS = {"gender": "string", "zip_code": "string", "title": "string", "genres": "string"}


class ValidationError(ValueError):
    """Raised when the source data breaks the dataset contract."""


def parse_dat(
    data: bytes, name: str, columns: Sequence[str], encoding: str = SOURCE_ENCODING
) -> pd.DataFrame:
    """Parse one ``::`` delimited MovieLens file into string columns (no type coercion)."""
    rows = []
    for line_number, line in enumerate(data.decode(encoding).splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(DELIMITER)
        if len(fields) != len(columns):
            raise ValidationError(
                f"{name} line {line_number}: expected {len(columns)} '{DELIMITER}'-separated "
                f"fields, got {len(fields)}: {line[:80]!r}"
            )
        rows.append([field.strip() for field in fields])
    if not rows:
        raise ValidationError(f"{name} contains no rows.")
    return pd.DataFrame(rows, columns=list(columns), dtype="string")


def _fail_if(mask: pd.Series, frame: pd.DataFrame, message: str) -> None:
    if mask.any():
        examples = frame.loc[mask].head(3).to_dict("records")
        raise ValidationError(f"{message}: {int(mask.sum())} row(s), e.g. {examples}")


def _integers(frame: pd.DataFrame, column: str, name: str, minimum: int, maximum=None):
    is_digits = frame[column].str.fullmatch(r"\d+").fillna(False).astype(bool)
    _fail_if(~is_digits, frame, f"{name}: '{column}' must be an integer")
    values = pd.to_numeric(frame[column]).astype("Int64")
    out_of_range = values.lt(minimum)
    if maximum is not None:
        out_of_range |= values.gt(maximum)
    bounds = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
    _fail_if(out_of_range, frame, f"{name}: '{column}' must be {bounds}")
    return values


def _require_text(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        _fail_if(frame[column].fillna("").eq(""), frame, f"{name}: '{column}' is empty")


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    _fail_if(frame.duplicated(list(columns), keep=False), frame, f"{name}: duplicate {columns}")


def validate_tables(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame):
    """Type and validate the three tables. Returns typed copies; raises ValidationError."""
    users, movies, ratings = users.copy(), movies.copy(), ratings.copy()

    users["user_id"] = _integers(users, "user_id", "users.dat", minimum=1)
    users["age_code"] = _integers(users, "age_code", "users.dat", minimum=0)
    users["occupation_code"] = _integers(users, "occupation_code", "users.dat", minimum=0)
    _require_text(users, ["gender", "zip_code"], "users.dat")
    _require_unique(users, ["user_id"], "users.dat")

    movies["movie_id"] = _integers(movies, "movie_id", "movies.dat", minimum=1)
    _require_text(movies, ["title", "genres"], "movies.dat")
    _require_unique(movies, ["movie_id"], "movies.dat")

    ratings["user_id"] = _integers(ratings, "user_id", "ratings.dat", minimum=1)
    ratings["movie_id"] = _integers(ratings, "movie_id", "ratings.dat", minimum=1)
    ratings["rating"] = _integers(ratings, "rating", "ratings.dat", minimum=1, maximum=5)
    ratings["timestamp"] = _integers(ratings, "timestamp", "ratings.dat", minimum=1)
    times = pd.to_datetime(ratings["timestamp"], unit="s", utc=True, errors="coerce")
    invalid_time = times.isna() | times.gt(pd.Timestamp.now(tz="UTC"))
    _fail_if(invalid_time, ratings, "ratings.dat: 'timestamp' is not a valid past Unix time")
    _require_unique(ratings, ["user_id", "movie_id"], "ratings.dat")

    _fail_if(~ratings["user_id"].isin(users["user_id"]), ratings, "ratings.dat: unknown user_id")
    _fail_if(
        ~ratings["movie_id"].isin(movies["movie_id"]), ratings, "ratings.dat: unknown movie_id"
    )
    return users, movies, ratings


def build_outputs(sources: Mapping[str, bytes]) -> dict[str, pd.DataFrame]:
    """Parse, validate and join the sources keyed by ``users``/``movies``/``ratings``."""
    users = parse_dat(sources["users"], "users.dat", USER_COLUMNS)
    movies = parse_dat(sources["movies"], "movies.dat", MOVIE_COLUMNS)
    ratings = parse_dat(sources["ratings"], "ratings.dat", RATING_COLUMNS)
    users, movies, ratings = validate_tables(users, movies, ratings)

    joined = ratings.merge(movies, on="movie_id", how="inner", validate="many_to_one").merge(
        users, on="user_id", how="inner", validate="many_to_one"
    )
    if len(joined) != len(ratings):
        raise ValidationError("Join lost ratings; every rating must match a user and a movie.")

    return {
        "ratings.csv": _sorted(joined[OUTPUT_COLUMNS], ["user_id", "movie_id"]),
        "users.csv": _sorted(users[USER_COLUMNS], ["user_id"]),
        "movies.csv": _sorted(movies[MOVIE_COLUMNS], ["movie_id"]),
    }


def _sorted(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_csv(
        BytesIO(data),
        encoding="utf-8",
        dtype=STRING_COLUMNS,
        keep_default_na=False,
        na_values=[""],
    )


def check_readback(name: str, data: bytes, expected: pd.DataFrame) -> None:
    """Confirm the encoded CSV is non-empty, UTF-8, and reads back identically with Pandas."""
    data.decode("utf-8")
    frame = read_csv_bytes(data)
    if frame.empty:
        raise ValidationError(f"{name} is empty.")
    if list(frame.columns) != list(expected.columns):
        raise ValidationError(f"{name} columns {list(frame.columns)} != {list(expected.columns)}")
    if len(frame) != len(expected):
        raise ValidationError(f"{name} reads back {len(frame)} rows, expected {len(expected)}")
    if "zip_code" in frame and not frame["zip_code"].equals(expected["zip_code"]):
        raise ValidationError(f"{name}: zip codes changed on readback (leading zeros?)")


def pipeline_version() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def build_manifest(
    sources: Mapping[str, bytes],
    source_keys: Mapping[str, str],
    outputs: Mapping[str, tuple[bytes, pd.DataFrame]],
    processed_prefix: str,
    bucket: str | None,
    version: str,
    created_at: datetime | None = None,
) -> dict:
    created_at = created_at or datetime.now(UTC)
    return {
        "dataset": DATASET,
        "bucket": bucket,
        "processed_prefix": processed_prefix,
        "pipeline_version": version,
        "created_at_utc": created_at.astimezone(UTC).isoformat(timespec="seconds"),
        "sources": [
            {
                "key": source_keys[table],
                "size_bytes": len(sources[table]),
                "sha256": sha256_hex(sources[table]),
            }
            for table in SOURCE_FILES
        ],
        "outputs": [
            {
                "key": f"{processed_prefix}{name}",
                "size_bytes": len(data),
                "sha256": sha256_hex(data),
                "row_count": len(frame),
                "columns": list(frame.columns),
            }
            for name, (data, frame) in outputs.items()
        ],
    }


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def load_sources_from_storage(client, bucket: str, raw_prefix: str):
    raw_prefix = normalize_prefix(raw_prefix)
    keys = {table: f"{raw_prefix}{filename}" for table, filename in SOURCE_FILES.items()}
    return {table: download(client, bucket, key) for table, key in keys.items()}, keys


def load_sources_from_dir(source_dir: Path):
    paths = {table: source_dir / filename for table, filename in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {', '.join(missing)}")
    return {table: path.read_bytes() for table, path in paths.items()}, {
        table: path.as_posix() for table, path in paths.items()
    }


def prepare(
    sources: Mapping[str, bytes],
    source_keys: Mapping[str, str],
    output_dir: Path,
    processed_prefix: str = PROCESSED_PREFIX,
    bucket: str | None = None,
    version: str | None = None,
) -> dict:
    """Validate and write the outputs plus manifest locally. Nothing is uploaded here."""
    processed_prefix = normalize_prefix(processed_prefix)
    frames = build_outputs(sources)
    outputs = {}
    for name in OUTPUT_FILES:
        data = to_csv_bytes(frames[name])
        check_readback(name, data, frames[name])
        outputs[name] = (data, frames[name])

    manifest = build_manifest(
        sources, source_keys, outputs, processed_prefix, bucket, version or pipeline_version()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (data, _) in outputs.items():
        (output_dir / name).write_bytes(data)
    (output_dir / MANIFEST_NAME).write_bytes(manifest_bytes(manifest))
    return manifest


def publish(client, bucket: str, output_dir: Path, processed_prefix: str) -> None:
    """Upload the prepared outputs, manifest last. Refuses to touch an existing version."""
    processed_prefix = normalize_prefix(processed_prefix)
    names = [*OUTPUT_FILES, MANIFEST_NAME]
    existing = [name for name in names if head(client, bucket, f"{processed_prefix}{name}")]
    if existing:
        raise FileExistsError(
            f"s3://{bucket}/{processed_prefix} already contains {existing}; "
            "publish changed output under a new version prefix (e.g. v2/)."
        )
    for name in names:
        content_type = "application/json" if name == MANIFEST_NAME else "text/csv"
        data = (output_dir / name).read_bytes()
        upload_verified(client, bucket, f"{processed_prefix}{name}", data, content_type)


def verify_published(client, bucket: str, processed_prefix: str) -> dict:
    """Download the published dataset and check it against its manifest."""
    processed_prefix = normalize_prefix(processed_prefix)
    manifest_key = f"{processed_prefix}{MANIFEST_NAME}"
    raw_manifest = download(client, bucket, manifest_key)
    manifest = json.loads(raw_manifest.decode("utf-8"))
    for entry in manifest["outputs"]:
        verify_object(client, bucket, entry["key"], entry["sha256"], entry["size_bytes"])
        frame = read_csv_bytes(download(client, bucket, entry["key"]))
        if len(frame) != entry["row_count"] or list(frame.columns) != entry["columns"]:
            raise ValidationError(f"{entry['key']} row count or schema differs from manifest.")
    return {
        "bucket": bucket,
        "manifest_uri": f"s3://{bucket}/{manifest_key}",
        "manifest_sha256": sha256_hex(raw_manifest),
        "ratings_uri": f"s3://{bucket}/{processed_prefix}ratings.csv",
        "outputs": [
            {key: entry[key] for key in ("key", "sha256", "row_count")}
            for entry in manifest["outputs"]
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/movielens-1m/v1"))
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read the .dat files from a local directory instead of object storage "
        "(development only; cannot be combined with --publish).",
    )
    parser.add_argument("--raw-prefix", default=RAW_PREFIX)
    parser.add_argument("--processed-prefix", default=PROCESSED_PREFIX)
    parser.add_argument("--bucket", help="Defaults to B2_BUCKET.")
    parser.add_argument(
        "--publish", action="store_true", help="Upload and verify after validation."
    )
    args = parser.parse_args(argv)

    if args.source_dir and args.publish:
        parser.error("--publish requires reading the sources from object storage.")

    client = settings = None
    if args.source_dir:
        sources, source_keys = load_sources_from_dir(args.source_dir)
    else:
        settings = StorageSettings.from_env(bucket=args.bucket)
        client = make_client(settings)
        sources, source_keys = load_sources_from_storage(client, settings.bucket, args.raw_prefix)

    bucket = settings.bucket if settings else None
    manifest = prepare(sources, source_keys, args.output_dir, args.processed_prefix, bucket)
    summary = {"output_dir": str(args.output_dir), "outputs": manifest["outputs"]}

    if args.publish:
        publish(client, settings.bucket, args.output_dir, args.processed_prefix)
        summary = verify_published(client, settings.bucket, args.processed_prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Clean the movie ratings dataset into a processed CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("user_id", "movie_id", "rating")
OPTIONAL_INTEGER_COLUMNS = (
    "timestamp",
    "release_year",
    "age_code",
    "occupation_code",
)
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
RATING_MIN, RATING_MAX = 1, 5
USER_COLUMNS = ("gender", "age_code", "occupation_code", "zip_code")
MOVIE_COLUMNS = ("title", "genres", "release_year")


def find_default_input(root: Path) -> Path:
    candidates = (
        root / "compressed_data.csv",
        root / "compressed_data.csv.gz",
        root / "data" / "raw" / "Netflix_Dataset_Rating.csv",
        root / "data" / "raw" / "Netflix_Dataset_Rating.csv.gz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No input dataset found. Pass --input or place the dataset in one of:\n{searched}"
    )


def _temporary_file(directory: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=suffix,
        prefix=".cleaning-",
        dir=directory,
        delete=False,
    ) as temporary:
        return Path(temporary.name)


def _parse_integers(values: pd.Series) -> pd.Series:
    """Strictly parse integer strings ("4" ok, "4.0" / "abc" / "" -> <NA>), vectorized."""
    is_integer = values.str.fullmatch(INTEGER_PATTERN.pattern).fillna(False).astype(bool)
    return pd.to_numeric(values.where(is_integer), errors="coerce").astype("Int64")


def _harmonize_metadata(frame: pd.DataFrame, key: str, columns: Sequence[str]) -> int:
    """Give every row of an entity the metadata of its most recent row (in place).

    Returns how many entities had conflicting metadata.
    """
    present = [column for column in columns if column in frame.columns]
    if not present:
        return 0
    conflicting = frame.groupby(key)[present].nunique().gt(1).any(axis=1)
    if not conflicting.any():
        return 0
    ordered = frame
    if "timestamp" in frame.columns:
        order = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(-1)
        ordered = frame.iloc[order.argsort(kind="stable")]
    latest = ordered.groupby(key)[present].last()
    frame[present] = latest.loc[frame[key]].to_numpy()
    return int(conflicting.sum())


def _read_dataset(input_path: Path) -> pd.DataFrame:
    opener = gzip.open if input_path.suffix.lower() == ".gz" else open
    with opener(input_path, "rt", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)

    if not header:
        raise ValueError("Input CSV is empty or has no header.")
    if len(header) != len(set(header)):
        raise ValueError("Input CSV contains duplicate column names.")

    missing_columns = set(REQUIRED_COLUMNS) - set(header)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    return pd.read_csv(
        input_path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
        compression="infer",
        on_bad_lines="error",
    )


def _normalize_genres(value: str) -> str:
    genres = dict.fromkeys(genre.strip() for genre in value.split("|") if genre.strip())
    return "|".join(genres)


def clean_dataset(input_path: Path, output_path: Path, report_path: Path) -> dict:
    """Validate, normalize, and deduplicate ratings with a Pandas pipeline."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")
    if input_path in (output_path, report_path) or output_path == report_path:
        raise ValueError("Input, output, and report must be different files.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_temp: Path | None = None
    report_temp: Path | None = None

    try:
        frame = _read_dataset(input_path)
        frame = frame.apply(lambda column: column.str.strip())
        rows_read = len(frame)
        invalid_reasons: Counter[str] = Counter()
        invalid_optional_values: Counter[str] = Counter()

        parsed_integers: dict[str, pd.Series] = {}
        invalid_masks: list[pd.Series] = []
        for column in ("user_id", "movie_id"):
            parsed = _parse_integers(frame[column])
            invalid = parsed.isna() | parsed.le(0).fillna(False)
            invalid_reasons[f"invalid_{column}"] = int(invalid.sum())
            parsed_integers[column] = parsed
            invalid_masks.append(invalid)

        parsed_ratings = _parse_integers(frame["rating"])
        invalid_rating = parsed_ratings.isna()
        rating_out_of_range = ~invalid_rating & ~parsed_ratings.between(RATING_MIN, RATING_MAX)
        invalid_reasons["invalid_rating"] = int(invalid_rating.sum())
        invalid_reasons["rating_out_of_range"] = int(rating_out_of_range.sum())
        parsed_integers["rating"] = parsed_ratings
        invalid_masks.extend((invalid_rating, rating_out_of_range))

        invalid_rows = pd.Series(False, index=frame.index)
        for invalid_mask in invalid_masks:
            invalid_rows |= invalid_mask.fillna(False).astype(bool)
        frame = frame.loc[~invalid_rows].copy()

        for column, parsed in parsed_integers.items():
            frame[column] = parsed.loc[frame.index].astype("string")

        if "title" in frame.columns:
            frame["title"] = frame["title"].str.split().str.join(" ")
        if "genres" in frame.columns:
            frame["genres"] = frame["genres"].map(_normalize_genres)
        if "gender" in frame.columns:
            frame["gender"] = frame["gender"].str.upper()

        for column in OPTIONAL_INTEGER_COLUMNS:
            if column not in frame.columns:
                continue
            nonblank = frame[column].ne("")
            parsed = _parse_integers(frame[column])
            if column == "timestamp":
                # Epoch seconds must be positive and not in the future.
                implausible = parsed.le(0) | parsed.gt(int(time.time()))
                parsed = parsed.mask(implausible.fillna(False))
            invalid = nonblank & parsed.isna()
            invalid_optional_values[column] = int(invalid.sum())
            frame[column] = parsed.astype("string").fillna("")

        blank_metadata_values = {
            column: int(frame[column].eq("").sum())
            for column in ("title", "genres")
            if column in frame.columns
        }
        duplicates_removed = int(frame.duplicated().sum())
        frame = frame.drop_duplicates(keep="first")

        # A user may re-rate a movie: keep only the most recent rating per pair.
        if "timestamp" in frame.columns:
            order = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(-1)
            frame = frame.iloc[order.argsort(kind="stable")]
        pair_duplicates = int(frame.duplicated(["user_id", "movie_id"], keep="last").sum())
        frame = frame.drop_duplicates(["user_id", "movie_id"], keep="last").sort_index()

        harmonized = {
            entity: _harmonize_metadata(frame, key, columns)
            for entity, key, columns in (
                ("users", "user_id", USER_COLUMNS),
                ("movies", "movie_id", MOVIE_COLUMNS),
            )
        }

        output_temp = _temporary_file(output_path.parent, ".csv")
        frame.to_csv(output_temp, index=False, encoding="utf-8", lineterminator="\n")

        report = {
            "input": input_path.name,
            "output": output_path.name,
            "rows_read": rows_read,
            "rows_written": len(frame),
            "invalid_rows_removed": int(invalid_rows.sum()),
            "exact_duplicates_removed": duplicates_removed,
            "older_duplicate_ratings_removed": pair_duplicates,
            "inconsistent_metadata_harmonized": harmonized,
            "invalid_row_reasons": dict(sorted(invalid_reasons.items())),
            "invalid_optional_values_cleared": dict(sorted(invalid_optional_values.items())),
            "blank_metadata_values": dict(sorted(blank_metadata_values.items())),
        }

        report_temp = _temporary_file(report_path.parent, ".json")
        with report_temp.open("w", encoding="utf-8", newline="") as destination:
            json.dump(report, destination, indent=2, ensure_ascii=False)
            destination.write("\n")

        os.replace(output_temp, output_path)
        output_temp = None
        os.replace(report_temp, report_path)
        report_temp = None
        return report
    finally:
        for temporary in (output_temp, report_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Clean and validate the movie ratings dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Input CSV or CSV.GZ (auto-detected from the repository if omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "processed" / "ratings_clean.csv",
        help="Cleaned CSV destination.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "data" / "processed" / "cleaning_report.json",
        help="JSON cleaning report destination.",
    )
    args = parser.parse_args(argv)

    input_path = args.input if args.input is not None else find_default_input(root)
    report = clean_dataset(input_path, args.output, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

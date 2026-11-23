import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.data import check_b2, prepare_movielens, upload_raw
from src.data.cleaner import clean_dataset
from src.data.prepare_movielens import (
    OUTPUT_COLUMNS,
    ValidationError,
    build_outputs,
    load_sources_from_storage,
    prepare,
    publish,
    read_csv_bytes,
    to_csv_bytes,
    verify_published,
)
from src.data.storage import StorageSettings, sha256_hex

BUCKET = "test-bucket"
PREFIX = "processed/movielens-1m/v1/"
USERS = ["1::F::1::10::02138", "2::M::56::16::70072", "3::M::25::15::55117"]
MOVIES = [
    "1::Toy Story (1995)::Animation|Children's|Comedy",
    "2::Misérables, Les (1995)::Drama|Musical",
    "3::Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)::Sci-Fi|War",
    '4::"Great" Escape, The (1963)::Adventure|War',
]
RATINGS = [
    "1::1::5::978300760",
    "1::2::3::978302109",
    "2::3::4::978298413",
    "3::4::1::978220179",
    "3::1::2::978199279",
]


def encode(lines):
    return ("\n".join(lines) + "\n").encode("latin-1")


def sources(users=USERS, movies=MOVIES, ratings=RATINGS):
    return {"users": encode(users), "movies": encode(movies), "ratings": encode(ratings)}


KEYS = {table: f"raw/movielens-1m/{table}.dat" for table in ("users", "movies", "ratings")}


def test_parses_joins_and_preserves_ids_and_zip_codes():
    outputs = build_outputs(sources())
    ratings = outputs["ratings.csv"]

    assert list(ratings.columns) == OUTPUT_COLUMNS
    assert len(ratings) == len(RATINGS)
    assert ratings[["user_id", "movie_id"]].values.tolist() == [
        [1, 1],
        [1, 2],
        [2, 3],
        [3, 1],
        [3, 4],
    ]
    first = ratings.iloc[0]
    assert (first["gender"], first["age_code"], first["zip_code"]) == ("F", 1, "02138")
    assert len(outputs["users.csv"]) == 3
    assert len(outputs["movies.csv"]) == 4


def test_titles_with_punctuation_and_accents_round_trip_as_utf8():
    data = to_csv_bytes(build_outputs(sources())["movies.csv"])
    titles = read_csv_bytes(data)["title"].tolist()

    assert titles == [line.split("::")[1] for line in MOVIES]
    assert "Misérables".encode() in data


def test_zip_codes_keep_leading_zeros_in_csv():
    data = to_csv_bytes(build_outputs(sources())["ratings.csv"])
    assert read_csv_bytes(data)["zip_code"].iloc[0] == "02138"


@pytest.mark.parametrize(
    "users",
    [
        [*USERS[:2], "3::M::25::15"],  # missing field
        [*USERS[:2], "3::M::25::15::55117::extra"],  # extra field
        [*USERS[:2], "3,M,25,15,55117"],  # wrong delimiter
        [*USERS[:2], "3:M:25:15:55117"],  # single-colon delimiter
    ],
)
def test_rejects_malformed_rows_with_line_number(users):
    with pytest.raises(ValidationError, match="users.dat line 3"):
        build_outputs(sources(users=users))


def test_rejects_duplicate_user_ids():
    with pytest.raises(ValidationError, match="duplicate"):
        build_outputs(sources(users=[*USERS, "3::F::18::4::10001"]))


def test_rejects_duplicate_movie_ids():
    with pytest.raises(ValidationError, match="duplicate"):
        build_outputs(sources(movies=[*MOVIES, "4::Other (2000)::Drama"]))


def test_rejects_duplicate_ratings():
    with pytest.raises(ValidationError, match="duplicate"):
        build_outputs(sources(ratings=[*RATINGS, "1::1::4::978300999"]))


@pytest.mark.parametrize(
    ("rating", "message"),
    [("99::1::5::978300760", "unknown user_id"), ("1::99::5::978300760", "unknown movie_id")],
)
def test_rejects_ratings_without_matching_user_or_movie(rating, message):
    with pytest.raises(ValidationError, match=message):
        build_outputs(sources(ratings=[*RATINGS, rating]))


@pytest.mark.parametrize("value", ["0", "6", "4.5", "-1", "abc", ""])
def test_rejects_invalid_ratings(value):
    with pytest.raises(ValidationError, match="'rating'"):
        build_outputs(sources(ratings=[*RATINGS, f"2::1::{value}::978300760"]))


@pytest.mark.parametrize("value", ["0", "abc", "", "99999999999999999"])
def test_rejects_invalid_timestamps(value):
    with pytest.raises(ValidationError, match="'timestamp'"):
        build_outputs(sources(ratings=[*RATINGS, f"2::1::4::{value}"]))


@pytest.mark.parametrize(
    ("table", "line"), [("users", "0::F::1::10::02138"), ("movies", "-4::Bad (2000)::Drama")]
)
def test_rejects_non_positive_dimension_ids(table, line):
    rows = {"users": USERS, "movies": MOVIES}[table]
    with pytest.raises(ValidationError):
        build_outputs(sources(**{table: [*rows, line]}))


def test_output_is_deterministic_regardless_of_source_order():
    first = build_outputs(sources())
    shuffled = build_outputs(sources(users=USERS[::-1], movies=MOVIES[::-1], ratings=RATINGS[::-1]))
    for name in first:
        assert to_csv_bytes(first[name]) == to_csv_bytes(shuffled[name])


def test_prepare_writes_files_and_manifest_with_checksums(tmp_path):
    created = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
    manifest = prepare_movielens.build_manifest(
        sources(),
        KEYS,
        {"ratings.csv": (b"abc", pd.DataFrame({"a": [1]}))},
        PREFIX,
        BUCKET,
        "abc123",
        created_at=created,
    )
    assert manifest["created_at_utc"] == "2026-11-21T12:00:00+00:00"
    assert manifest["outputs"][0]["sha256"] == sha256_hex(b"abc")

    manifest = prepare(sources(), KEYS, tmp_path, PREFIX, BUCKET, version="abc123")
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert saved == manifest
    assert saved["pipeline_version"] == "abc123"
    assert [s["key"] for s in saved["sources"]] == list(KEYS.values())
    assert saved["sources"][0]["sha256"] == sha256_hex(sources()["users"])
    for entry in saved["outputs"]:
        data = (tmp_path / entry["key"].removeprefix(PREFIX)).read_bytes()
        assert entry["sha256"] == sha256_hex(data)
        assert entry["size_bytes"] == len(data)
    ratings_entry = saved["outputs"][0]
    assert ratings_entry["key"] == f"{PREFIX}ratings.csv"
    assert ratings_entry["row_count"] == len(RATINGS)
    assert ratings_entry["columns"] == OUTPUT_COLUMNS


def test_generated_csv_is_accepted_by_existing_cleaner(tmp_path):
    prepare(sources(), KEYS, tmp_path, PREFIX, version="test")
    report = clean_dataset(
        tmp_path / "ratings.csv", tmp_path / "clean.csv", tmp_path / "report.json"
    )
    assert report["rows_written"] == len(RATINGS)


def test_reads_sources_from_storage_prefix(s3):
    for table, key in KEYS.items():
        s3.put_object(Bucket=BUCKET, Key=key, Body=sources()[table])

    loaded, keys = load_sources_from_storage(s3, BUCKET, "raw/movielens-1m")

    assert keys == KEYS
    assert loaded == sources()


def test_publish_uploads_manifest_last_and_verifies(s3, tmp_path):
    prepare(sources(), KEYS, tmp_path, PREFIX, BUCKET, version="test")
    publish(s3, BUCKET, tmp_path, PREFIX)
    result = verify_published(s3, BUCKET, PREFIX)

    assert s3.put_order[-1] == f"{PREFIX}manifest.json"
    assert result["ratings_uri"] == f"s3://{BUCKET}/{PREFIX}ratings.csv"
    assert result["manifest_sha256"] == sha256_hex((tmp_path / "manifest.json").read_bytes())


def test_publish_refuses_to_overwrite_existing_version(s3, tmp_path):
    prepare(sources(), KEYS, tmp_path, PREFIX, BUCKET, version="test")
    publish(s3, BUCKET, tmp_path, PREFIX)
    with pytest.raises(FileExistsError, match="new version prefix"):
        publish(s3, BUCKET, tmp_path, PREFIX)


def test_verify_detects_tampered_object(s3, tmp_path):
    prepare(sources(), KEYS, tmp_path, PREFIX, BUCKET, version="test")
    publish(s3, BUCKET, tmp_path, PREFIX)
    s3.objects[(BUCKET, f"{PREFIX}ratings.csv")] += b"1,1,5,1,x,y,F,1,1,00000\n"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_published(s3, BUCKET, PREFIX)


def test_upload_raw_uploads_once_and_never_overwrites(s3, tmp_path):
    for table, data in sources().items():
        (tmp_path / f"{table}.dat").write_bytes(data)

    first = upload_raw.upload_raw_files(s3, BUCKET, tmp_path)
    second = upload_raw.upload_raw_files(s3, BUCKET, tmp_path)

    assert [r["status"] for r in first] == ["uploaded"] * 3
    assert [r["status"] for r in second] == ["already present, verified"] * 3
    assert first[2]["sha256"] == sha256_hex(sources()["ratings"])

    (tmp_path / "ratings.dat").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        upload_raw.upload_raw_files(s3, BUCKET, tmp_path)
    assert s3.objects[(BUCKET, KEYS["ratings"])] == sources()["ratings"]


def test_roundtrip_writes_test_object_under_test_prefix(s3):
    result = check_b2.roundtrip(s3, BUCKET, now=datetime(2026, 11, 21, tzinfo=UTC))
    assert result["key"] == "test/noe/roundtrip-20261121T000000Z.txt"
    assert result["listed"] is True


def test_storage_settings_require_environment(monkeypatch):
    for name in ("B2_ENDPOINT", "B2_REGION", "B2_BUCKET", "B2_KEY_ID", "B2_APPLICATION_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="B2_ENDPOINT"):
        StorageSettings.from_env()

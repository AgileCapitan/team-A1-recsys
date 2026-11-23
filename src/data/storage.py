"""Shared S3-compatible object storage helpers (Backblaze B2) for the data pipeline.

Connection settings come from environment variables only, never from code:

- ``B2_ENDPOINT``: explicit S3 endpoint URL, e.g. ``https://s3.us-east-005.backblazeb2.com``
- ``B2_REGION``: e.g. ``us-east-005``
- ``B2_BUCKET``: bucket name
- ``B2_KEY_ID`` / ``B2_APPLICATION_KEY``: application key credentials

Any S3-compatible service (RustFS, MinIO...) works by pointing ``B2_ENDPOINT`` at it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

RAW_PREFIX = "raw/movielens-1m/"
PROCESSED_PREFIX = "processed/movielens-1m/v1/"
TEST_PREFIX = "test/noe/"
MISSING_OBJECT_CODES = {"404", "NoSuchKey", "NotFound"}


@dataclass(frozen=True)
class StorageSettings:
    endpoint: str
    region: str
    bucket: str
    key_id: str
    application_key: str

    @classmethod
    def from_env(cls, bucket: str | None = None) -> StorageSettings:
        values = {
            "B2_ENDPOINT": os.getenv("B2_ENDPOINT"),
            "B2_REGION": os.getenv("B2_REGION"),
            "B2_BUCKET": bucket or os.getenv("B2_BUCKET"),
            "B2_KEY_ID": os.getenv("B2_KEY_ID"),
            "B2_APPLICATION_KEY": os.getenv("B2_APPLICATION_KEY"),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing storage environment variables: {', '.join(missing)}")
        return cls(
            endpoint=values["B2_ENDPOINT"],
            region=values["B2_REGION"],
            bucket=values["B2_BUCKET"],
            key_id=values["B2_KEY_ID"],
            application_key=values["B2_APPLICATION_KEY"],
        )


def make_client(settings: StorageSettings):
    """Create a boto3 S3 client bound to the explicit endpoint URL."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        region_name=settings.region,
        aws_access_key_id=settings.key_id,
        aws_secret_access_key=settings.application_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip().lstrip("/")
    if not prefix:
        raise ValueError("Object prefix must not be empty.")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _is_missing(error: Exception) -> bool:
    response = getattr(error, "response", None) or {}
    return str(response.get("Error", {}).get("Code")) in MISSING_OBJECT_CODES


def head(client, bucket: str, key: str) -> dict | None:
    """Return object metadata, or ``None`` if the object does not exist."""
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        if _is_missing(error):
            return None
        raise


def download(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def upload_verified(
    client, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"
) -> dict:
    """Upload ``data`` without overwriting, then read it back and verify size and SHA-256."""
    if head(client, bucket, key) is not None:
        raise FileExistsError(f"s3://{bucket}/{key} already exists; objects are immutable.")
    digest = sha256_hex(data)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        Metadata={"sha256": digest},
    )
    return verify_object(client, bucket, key, expected_sha256=digest, expected_size=len(data))


def verify_object(
    client, bucket: str, key: str, expected_sha256: str, expected_size: int | None = None
) -> dict:
    data = download(client, bucket, key)
    digest = sha256_hex(data)
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {key}: expected {expected_sha256}, got {digest}")
    if expected_size is not None and len(data) != expected_size:
        raise ValueError(f"Size mismatch for {key}: expected {expected_size}, got {len(data)}")
    return {"key": key, "size_bytes": len(data), "sha256": digest}

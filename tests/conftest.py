"""Shared fixtures: an in-memory S3 client so tests never need live object storage."""

import io

import pytest


class FakeClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """Minimal in-memory stand-in for the boto3 S3 client methods the pipeline uses."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_order: list[str] = []

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = bytes(Body)
        self.put_order.append(Key)
        return {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def list_objects_v2(self, Bucket, Prefix=""):
        keys = sorted(key for bucket, key in self.objects if bucket == Bucket)
        return {"Contents": [{"Key": key} for key in keys if key.startswith(Prefix)]}


@pytest.fixture
def s3():
    return FakeS3()

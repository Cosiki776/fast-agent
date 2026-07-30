from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from fast_agent.transactional.storage import artifact_store as artifact_store_module
from fast_agent.transactional.storage.artifact_store import (
    ArtifactCorruptionError,
    ArtifactId,
    ArtifactKind,
    ArtifactNotFoundError,
    FileArtifactStore,
    InvalidArtifactIdError,
    artifact_id_for,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_put_read_and_reopen_preserve_complete_artifact(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    content = '{"stdout":"完成","exit_code":0}\n'.encode()
    expected_digest = hashlib.sha256(content).hexdigest()

    store = FileArtifactStore(root)
    metadata = store.put(
        content,
        media_type="application/json",
        kind=ArtifactKind.RAW_RESULT,
    )

    assert metadata.artifact_id == ArtifactId(f"sha256:{expected_digest}")
    assert metadata.sha256 == expected_digest
    assert metadata.byte_size == len(content)
    assert metadata.media_types == ("application/json",)
    assert metadata.kinds == (ArtifactKind.RAW_RESULT,)
    assert store.contains(metadata.artifact_id)
    assert store.read(metadata.artifact_id) == content

    reopened = FileArtifactStore(root)
    assert reopened.metadata(metadata.artifact_id) == metadata
    assert reopened.read(metadata.artifact_id) == content


def test_same_content_reuses_id_and_accumulates_usage_metadata(tmp_path: Path) -> None:
    content = b"shared tool output"
    store = FileArtifactStore(tmp_path / "artifacts")

    stdout_metadata = store.put(
        content,
        media_type="text/plain",
        kind=ArtifactKind.STDOUT,
    )
    raw_metadata = store.put(
        content,
        media_type="application/octet-stream",
        kind=ArtifactKind.RAW_RESULT,
    )

    assert stdout_metadata.artifact_id == raw_metadata.artifact_id
    assert raw_metadata.media_types == ("application/octet-stream", "text/plain")
    assert raw_metadata.kinds == (ArtifactKind.RAW_RESULT, ArtifactKind.STDOUT)
    assert store.metadata(raw_metadata.artifact_id) == raw_metadata


def test_different_content_has_different_artifact_ids(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")

    stdout = store.put(
        b"stdout",
        media_type="text/plain",
        kind=ArtifactKind.STDOUT,
    )
    stderr = store.put(
        b"stderr",
        media_type="text/plain",
        kind=ArtifactKind.STDERR,
    )

    assert stdout.artifact_id != stderr.artifact_id
    assert artifact_id_for(b"stdout") == stdout.artifact_id
    assert artifact_id_for(b"stderr") == stderr.artifact_id


def test_failed_metadata_publish_does_not_expose_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    content = b"complete evidence"
    artifact_id = artifact_id_for(content)
    real_replace = artifact_store_module.os.replace
    replace_calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated metadata publish failure")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_store_module.os, "replace", fail_second_replace)
        with pytest.raises(OSError, match="simulated metadata publish failure"):
            store.put(
                content,
                media_type="application/octet-stream",
                kind=ArtifactKind.RAW_RESULT,
            )

    assert not store.contains(artifact_id)
    with pytest.raises(ArtifactNotFoundError):
        store.metadata(artifact_id)

    metadata = store.put(
        content,
        media_type="application/octet-stream",
        kind=ArtifactKind.RAW_RESULT,
    )
    assert store.contains(metadata.artifact_id)
    assert store.read(metadata.artifact_id) == content


def test_read_detects_corrupted_content(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    metadata = store.put(
        b"original content",
        media_type="text/plain",
        kind=ArtifactKind.STDOUT,
    )
    object_path = root / "objects" / metadata.sha256[:2] / metadata.sha256
    object_path.write_bytes(b"tampered")

    with pytest.raises(ArtifactCorruptionError, match="size does not match"):
        store.read(metadata.artifact_id)


def test_invalid_artifact_id_cannot_escape_store_root(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")

    with pytest.raises(
        InvalidArtifactIdError,
        match="must contain a lowercase SHA-256 digest",
    ):
        store.read(ArtifactId("sha256:../../outside"))


def test_put_rejects_empty_media_type(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="media type must not be empty"):
        store.put(
            b"content",
            media_type="",
            kind=ArtifactKind.RAW_RESULT,
        )

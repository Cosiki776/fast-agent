from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from collections.abc import Mapping

ArtifactId = NewType("ArtifactId", str)

_ARTIFACT_ID_PREFIX = "sha256:"
_SHA256_HEX_LENGTH = 64


class ArtifactKind(StrEnum):
    """Semantic source of bytes stored as transactional evidence."""

    RAW_RESULT = "raw_result"
    STDOUT = "stdout"
    STDERR = "stderr"
    WORKSPACE_PATCH = "workspace_patch"


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Validated metadata associated with one content-addressed artifact."""

    artifact_id: ArtifactId
    sha256: str
    byte_size: int
    media_types: tuple[str, ...]
    kinds: tuple[ArtifactKind, ...]


class ArtifactStoreError(RuntimeError):
    """Base error raised by the file artifact store."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact is not completely available."""


class ArtifactCorruptionError(ArtifactStoreError):
    """Raised when persisted content or metadata fails validation."""


class InvalidArtifactIdError(ArtifactStoreError):
    """Raised when an artifact ID is not a canonical SHA-256 identifier."""


class FileArtifactStore:
    """Atomic content-addressed storage for complete raw tool evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._objects_root = root / "objects"
        self._metadata_root = root / "metadata"
        self._objects_root.mkdir(parents=True, exist_ok=True)
        self._metadata_root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        content: bytes,
        *,
        media_type: str,
        kind: ArtifactKind,
    ) -> ArtifactMetadata:
        """Store complete bytes before atomically publishing their metadata."""
        if not media_type:
            raise ValueError("Artifact media type must not be empty")

        artifact_id = artifact_id_for(content)
        digest = _digest_from_id(artifact_id)
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            sha256=digest,
            byte_size=len(content),
            media_types=(media_type,),
            kinds=(kind,),
        )
        object_path, metadata_path = self._paths_for(artifact_id)

        if metadata_path.exists():
            existing = self.metadata(artifact_id)
            if existing.byte_size != len(content):
                raise ArtifactCorruptionError(
                    f"Artifact '{artifact_id}' size does not match its content address"
                )
            metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                sha256=digest,
                byte_size=len(content),
                media_types=tuple(sorted({*existing.media_types, media_type})),
                kinds=tuple(sorted({*existing.kinds, kind}, key=lambda item: item.value)),
            )

        _store_content(object_path, content, digest)
        if not metadata_path.exists() or self.metadata(artifact_id) != metadata:
            _atomic_write(metadata_path, _encode_metadata(metadata))
        return metadata

    def read(self, artifact_id: ArtifactId) -> bytes:
        """Read and validate a completely published artifact."""
        metadata = self.metadata(artifact_id)
        object_path, _ = self._paths_for(artifact_id)
        if not object_path.is_file():
            raise ArtifactNotFoundError(f"Artifact content '{artifact_id}' was not found")

        content = object_path.read_bytes()
        if len(content) != metadata.byte_size:
            raise ArtifactCorruptionError(
                f"Artifact '{artifact_id}' size does not match its metadata"
            )
        digest = hashlib.sha256(content).hexdigest()
        if digest != metadata.sha256:
            raise ArtifactCorruptionError(
                f"Artifact '{artifact_id}' content hash does not match its ID"
            )
        return content

    def metadata(self, artifact_id: ArtifactId) -> ArtifactMetadata:
        """Read and validate metadata for a published artifact."""
        _, metadata_path = self._paths_for(artifact_id)
        if not metadata_path.is_file():
            raise ArtifactNotFoundError(f"Artifact metadata '{artifact_id}' was not found")
        return _decode_metadata(metadata_path.read_bytes(), expected_id=artifact_id)

    def contains(self, artifact_id: ArtifactId) -> bool:
        """Return whether both content and metadata are atomically visible."""
        object_path, metadata_path = self._paths_for(artifact_id)
        return object_path.is_file() and metadata_path.is_file()

    def _paths_for(self, artifact_id: ArtifactId) -> tuple[Path, Path]:
        digest = _digest_from_id(artifact_id)
        shard = digest[:2]
        return (
            self._objects_root / shard / digest,
            self._metadata_root / shard / f"{digest}.json",
        )


def artifact_id_for(content: bytes) -> ArtifactId:
    """Return the canonical content address for a byte sequence."""
    return ArtifactId(f"{_ARTIFACT_ID_PREFIX}{hashlib.sha256(content).hexdigest()}")


def _store_content(path: Path, content: bytes, expected_digest: str) -> None:
    if path.exists():
        existing_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing_digest != expected_digest:
            raise ArtifactCorruptionError(
                f"Artifact object '{expected_digest}' contains different bytes"
            )
        return
    _atomic_write(path, content)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _encode_metadata(metadata: ArtifactMetadata) -> bytes:
    payload = asdict(metadata)
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_metadata(
    content: bytes,
    *,
    expected_id: ArtifactId,
) -> ArtifactMetadata:
    try:
        raw_payload: object = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCorruptionError("Artifact metadata is not valid JSON") from exc
    payload = _metadata_object(raw_payload)

    artifact_id = ArtifactId(_metadata_str(payload, "artifact_id"))
    digest = _metadata_str(payload, "sha256")
    byte_size = _metadata_int(payload, "byte_size")
    media_types = _metadata_strings(payload, "media_types")
    kind_values = _metadata_strings(payload, "kinds")
    try:
        kinds = tuple(ArtifactKind(value) for value in kind_values)
    except ValueError as exc:
        raise ArtifactCorruptionError("Artifact metadata contains an unknown kind") from exc

    if artifact_id != expected_id:
        raise ArtifactCorruptionError(
            f"Artifact metadata ID '{artifact_id}' does not match '{expected_id}'"
        )
    expected_digest = _digest_from_id(expected_id)
    if digest != expected_digest:
        raise ArtifactCorruptionError(
            f"Artifact metadata hash '{digest}' does not match '{expected_digest}'"
        )
    if byte_size < 0:
        raise ArtifactCorruptionError("Artifact metadata byte size must not be negative")
    if not media_types or any(not media_type for media_type in media_types):
        raise ArtifactCorruptionError("Artifact metadata must contain non-empty media types")
    if not kinds:
        raise ArtifactCorruptionError("Artifact metadata must contain artifact kinds")

    return ArtifactMetadata(
        artifact_id=artifact_id,
        sha256=digest,
        byte_size=byte_size,
        media_types=media_types,
        kinds=kinds,
    )


def _digest_from_id(artifact_id: ArtifactId) -> str:
    value = str(artifact_id)
    if not value.startswith(_ARTIFACT_ID_PREFIX):
        raise InvalidArtifactIdError(f"Artifact ID '{value}' must start with 'sha256:'")
    digest = value.removeprefix(_ARTIFACT_ID_PREFIX)
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise InvalidArtifactIdError(
            f"Artifact ID '{value}' must contain a lowercase SHA-256 digest"
        )
    return digest


def _metadata_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactCorruptionError("Artifact metadata must be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ArtifactCorruptionError("Artifact metadata must contain only string keys")
        result[key] = item
    return result


def _metadata_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ArtifactCorruptionError(f"Artifact metadata field '{key}' must be a string")
    return value


def _metadata_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactCorruptionError(f"Artifact metadata field '{key}' must be an integer")
    return value


def _metadata_strings(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ArtifactCorruptionError(f"Artifact metadata field '{key}' must be a string list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ArtifactCorruptionError(f"Artifact metadata field '{key}' must be a string list")
        result.append(item)
    return tuple(result)

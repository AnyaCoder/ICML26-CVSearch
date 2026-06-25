from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


ArtifactKind = Literal["image", "json"]


@dataclass(frozen=True)
class ArtifactRecord:
    """A single saved debug artifact indexed in the sample manifest."""

    stage: str
    name: str
    kind: ArtifactKind
    path: str
    description: str | None = None
    target_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactStore:
    """Central writer for per-sample debug artifacts and their manifest."""

    root: str | Path
    manifest_name: str = "artifact_manifest.json"
    image_quality: int = 88

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records: list[ArtifactRecord] = []
        self._counters: dict[tuple[str, str], int] = {}
        self._load_existing_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_name

    def image(
        self,
        stage: str,
        name: str,
        image: Any,
        *,
        description: str | None = None,
        target_id: str | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        ordinal: int | None = None,
    ) -> Path:
        path = self._path(stage, name, suffix=".jpg", ordinal=ordinal, kind="image")
        return self.image_at(
            path.relative_to(self.root),
            stage,
            name,
            image,
            description=description,
            target_id=target_id,
            source_id=source_id,
            metadata=metadata,
        )

    def image_at(
        self,
        relative_path: str | Path,
        stage: str,
        name: str,
        image: Any,
        *,
        description: str | None = None,
        target_id: str | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        from .image_io import save_debug_image

        saved_path = save_debug_image(image, self.root / relative_path, quality=self.image_quality)
        self._record(
            ArtifactRecord(
                stage=stage,
                name=name,
                kind="image",
                path=str(saved_path.relative_to(self.root)),
                description=description,
                target_id=target_id,
                source_id=source_id,
                metadata=metadata or {},
            )
        )
        return saved_path

    def json(
        self,
        stage: str,
        name: str,
        data: Any,
        *,
        description: str | None = None,
        target_id: str | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        ordinal: int | None = None,
    ) -> Path:
        path = self._path(stage, name, suffix=".json", ordinal=ordinal, kind="json")
        return self.json_at(
            path.relative_to(self.root),
            stage,
            name,
            data,
            description=description,
            target_id=target_id,
            source_id=source_id,
            metadata=metadata,
        )

    def json_at(
        self,
        relative_path: str | Path,
        stage: str,
        name: str,
        data: Any,
        *,
        description: str | None = None,
        target_id: str | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))
        self._record(
            ArtifactRecord(
                stage=stage,
                name=name,
                kind="json",
                path=str(path.relative_to(self.root)),
                description=description,
                target_id=target_id,
                source_id=source_id,
                metadata=metadata or {},
            )
        )
        return path

    def records(self) -> list[ArtifactRecord]:
        return list(self._records)

    def existing(
        self,
        relative_path: str | Path,
        stage: str,
        name: str,
        *,
        kind: ArtifactKind = "image",
        description: str | None = None,
        target_id: str | None = None,
        source_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.root / relative_path
        self._record(
            ArtifactRecord(
                stage=stage,
                name=name,
                kind=kind,
                path=str(Path(relative_path)),
                description=description,
                target_id=target_id,
                source_id=source_id,
                metadata=metadata or {},
            )
        )
        return path

    def write_manifest(self) -> Path:
        self.manifest_path.write_text(json.dumps(to_jsonable(self._records), indent=2, ensure_ascii=False))
        return self.manifest_path

    def _path(self, stage: str, name: str, *, suffix: str, ordinal: int | None, kind: ArtifactKind) -> Path:
        if ordinal is None:
            existing = self._existing_record(stage, name, kind)
            if existing is not None:
                return self.root / existing.path
        stage_slug = slug(stage)
        name_slug = slug(name)
        prefix = f"{ordinal:02d}_" if ordinal is not None and ordinal >= 0 else ""
        return self.root / stage_slug / f"{prefix}{name_slug}{suffix}"

    def _record(self, record: ArtifactRecord) -> None:
        key = (record.stage, record.name, record.kind)
        self._records = [
            existing
            for existing in self._records
            if (existing.stage, existing.name, existing.kind) != key
        ]
        self._records.append(record)
        self.write_manifest()

    def _existing_record(self, stage: str, name: str, kind: ArtifactKind) -> ArtifactRecord | None:
        for record in self._records:
            if record.stage == stage and record.name == name and record.kind == kind:
                return record
        return None

    def _load_existing_manifest(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            records = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError:
            return
        for item in records:
            if not isinstance(item, dict):
                continue
            try:
                record = ArtifactRecord(
                    stage=item["stage"],
                    name=item["name"],
                    kind=item["kind"],
                    path=item["path"],
                    description=item.get("description"),
                    target_id=item.get("target_id"),
                    source_id=item.get("source_id"),
                    metadata=item.get("metadata") or {},
                )
            except KeyError:
                continue
            self._records.append(record)
            suffix = Path(record.path).suffix
            if suffix:
                counter_key = (slug(record.stage), slug(record.name))
                self._counters[counter_key] = max(self._counters.get(counter_key, 0), 1)


def artifact_store_from_context(context: Mapping[str, Any] | None) -> ArtifactStore | None:
    if not context:
        return None
    store = context.get("artifact_store")
    return store if isinstance(store, ArtifactStore) else None


def slug(text: str, max_len: int = 80) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return (value or "artifact")[:max_len]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return round(value, 4)
    return value


__all__ = [
    "ArtifactKind",
    "ArtifactRecord",
    "ArtifactStore",
    "artifact_store_from_context",
    "slug",
    "to_jsonable",
]

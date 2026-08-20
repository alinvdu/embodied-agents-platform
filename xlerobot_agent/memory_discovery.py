# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any


ENVIRONMENT_MAP_FILENAME = "environment_map.json"
HOME_MEMORY_FILENAME = "home_memory.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class EnvironmentMemoryRecord:
    memory_id: str
    label: str
    directory: Path
    home_memory_path: Path | None = None
    environment_map_path: Path | None = None
    manifest_path: Path | None = None
    status: str = "unknown"
    updated_at: float = 0.0
    region_count: int = 0
    place_count: int = 0
    object_count: int = 0
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "label": self.label,
            "directory": str(self.directory),
            "home_memory_path": str(self.home_memory_path) if self.home_memory_path else None,
            "environment_map_path": str(self.environment_map_path) if self.environment_map_path else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "status": self.status,
            "updated_at": self.updated_at,
            "region_count": self.region_count,
            "place_count": self.place_count,
            "object_count": self.object_count,
            "approved": self.approved,
        }


class EnvironmentMemoryDiscovery:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def list(self) -> list[EnvironmentMemoryRecord]:
        records: list[EnvironmentMemoryRecord] = []
        if self.root.exists():
            for directory in self.root.iterdir():
                if directory.is_dir():
                    records.append(self._record_from_directory(directory))
        records.extend(self._legacy_records())
        return sorted(records, key=lambda item: (item.updated_at, item.memory_id), reverse=True)

    def get(self, memory_id: str) -> EnvironmentMemoryRecord | None:
        query = _slug(memory_id)
        for record in self.list():
            if _slug(record.memory_id) == query:
                return record
        return None

    def create(self, memory_id: str, *, label: str | None = None) -> EnvironmentMemoryRecord:
        normalized = _slug(memory_id or label or "home")
        directory = self.root / normalized
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "memory_id": normalized,
            "label": label or normalized.replace("_", " "),
            "status": "draft",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        manifest_path = directory / MANIFEST_FILENAME
        if manifest_path.exists():
            existing = _load_json(manifest_path) or {}
            manifest = {**existing, **manifest, "created_at": existing.get("created_at", manifest["created_at"])}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return self._record_from_directory(directory)

    def environment_map_path(self, memory_id: str) -> Path:
        return self.root / _slug(memory_id) / ENVIRONMENT_MAP_FILENAME

    def home_memory_path(self, memory_id: str) -> Path:
        return self.root / _slug(memory_id) / HOME_MEMORY_FILENAME

    def save_environment(
        self,
        *,
        memory: dict[str, Any],
        environment_snapshot: dict[str, Any],
        label: str | None = None,
    ) -> EnvironmentMemoryRecord:
        memory_id = _slug(str(memory.get("memory_id") or "home"))
        directory = self.root / memory_id
        directory.mkdir(parents=True, exist_ok=True)
        home_memory_path = directory / HOME_MEMORY_FILENAME
        environment_map_path = directory / ENVIRONMENT_MAP_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        home_memory_path.write_text(json.dumps(memory, indent=2, sort_keys=True))
        environment_map_path.write_text(json.dumps(environment_snapshot, indent=2, sort_keys=True))
        manifest = {
            "memory_id": memory_id,
            "label": label or str(memory.get("memory_id") or memory_id),
            "status": "approved" if memory.get("approved") else "draft",
            "updated_at": time.time(),
            "home_memory_path": str(home_memory_path),
            "environment_map_path": str(environment_map_path),
            "region_count": len(memory.get("regions", []) or []),
            "place_count": len(memory.get("places", []) or []),
            "object_count": len(memory.get("objects", []) or []),
            "approved": bool(memory.get("approved", False)),
        }
        if manifest_path.exists():
            existing = _load_json(manifest_path) or {}
            manifest["created_at"] = existing.get("created_at", time.time())
        else:
            manifest["created_at"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return self._record_from_directory(directory)

    def load_environment_snapshot(self, memory_id: str) -> dict[str, Any] | None:
        record = self.get(memory_id)
        if record is None or record.environment_map_path is None:
            return None
        return _load_json(record.environment_map_path)

    def _record_from_directory(self, directory: Path) -> EnvironmentMemoryRecord:
        manifest_path = directory / MANIFEST_FILENAME
        home_memory_path = directory / HOME_MEMORY_FILENAME
        environment_map_path = directory / ENVIRONMENT_MAP_FILENAME
        manifest = _load_json(manifest_path) if manifest_path.exists() else {}
        memory = _load_json(home_memory_path) if home_memory_path.exists() else {}
        memory_id = str(manifest.get("memory_id") or memory.get("memory_id") or directory.name)
        label = str(manifest.get("label") or memory_id)
        updated_at = float(
            manifest.get("updated_at")
            or memory.get("updated_at")
            or _latest_mtime([manifest_path, home_memory_path, environment_map_path])
            or 0.0
        )
        regions = memory.get("regions", []) if isinstance(memory, dict) else []
        places = memory.get("places", []) if isinstance(memory, dict) else []
        objects = memory.get("objects", []) if isinstance(memory, dict) else []
        return EnvironmentMemoryRecord(
            memory_id=memory_id,
            label=label,
            directory=directory,
            home_memory_path=home_memory_path if home_memory_path.exists() else None,
            environment_map_path=environment_map_path if environment_map_path.exists() else None,
            manifest_path=manifest_path if manifest_path.exists() else None,
            status=str(manifest.get("status") or ("approved" if memory.get("approved") else "draft")),
            updated_at=updated_at,
            region_count=int(manifest.get("region_count") or len(regions or [])),
            place_count=int(manifest.get("place_count") or len(places or [])),
            object_count=int(manifest.get("object_count") or len(objects or [])),
            approved=bool(manifest.get("approved", memory.get("approved", False))),
        )

    def _legacy_records(self) -> list[EnvironmentMemoryRecord]:
        legacy_root = self.root.parent / "home_memory"
        if not legacy_root.exists():
            return []
        records: list[EnvironmentMemoryRecord] = []
        for path in legacy_root.glob("*.home_memory.json"):
            memory = _load_json(path) or {}
            memory_id = str(memory.get("memory_id") or path.stem.replace(".home_memory", ""))
            records.append(
                EnvironmentMemoryRecord(
                    memory_id=memory_id,
                    label=memory_id,
                    directory=path.parent,
                    home_memory_path=path,
                    environment_map_path=None,
                    manifest_path=None,
                    status="legacy",
                    updated_at=float(memory.get("updated_at") or path.stat().st_mtime),
                    region_count=len(memory.get("regions", []) or []),
                    place_count=len(memory.get("places", []) or []),
                    object_count=len(memory.get("objects", []) or []),
                    approved=bool(memory.get("approved", False)),
                )
            )
        return records


def default_memory_root_for_map_path(path: str | Path) -> Path:
    return Path(path).parent / "memories"


def default_environment_memory_dir_for_map_path(path: str | Path) -> Path:
    map_path = Path(path)
    memory_id = map_path.stem
    for suffix in ("_map", "_exploration", "_snapshot"):
        if memory_id.endswith(suffix):
            memory_id = memory_id[: -len(suffix)] or map_path.stem
            break
    return default_memory_root_for_map_path(map_path) / _slug(memory_id)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_mtime(paths: list[Path]) -> float:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else 0.0


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    return "_".join(part for part in cleaned.split("_") if part) or "home"

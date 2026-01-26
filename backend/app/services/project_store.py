"""Simple JSON-backed storage for saved research projects."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class ProjectStore:
    """Persist and load projects as JSON files on disk."""

    def __init__(self, base_dir: Path | None = None) -> None:
        root = base_dir or Path(__file__).resolve().parents[3]
        self.projects_dir = root / "data" / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _project_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{project_id}.json"

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for path in self.projects_dir.glob("*.json"):
            try:
                data = self._read_json(path)
            except json.JSONDecodeError:
                continue
            projects.append(
                {
                    "id": data.get("id", path.stem),
                    "name": data.get("name", "Untitled"),
                    "product_description": data.get("product_description", ""),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at", data.get("created_at")),
                }
            )
        projects.sort(key=lambda p: p.get("updated_at") or "", reverse=True)
        return projects

    def load_project(self, project_id: str) -> dict[str, Any]:
        path = self._project_path(project_id)
        if not path.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        return self._read_json(path)

    def save_project(self, project: dict[str, Any]) -> dict[str, Any]:
        project_id = project.get("id") or str(uuid4())
        path = self._project_path(project_id)

        now = datetime.now().isoformat(timespec="seconds")
        created_at = project.get("created_at")
        if path.exists():
            try:
                existing = self._read_json(path)
                created_at = existing.get("created_at", created_at)
            except json.JSONDecodeError:
                created_at = created_at or now
        if not created_at:
            created_at = now

        project["id"] = project_id
        project["created_at"] = created_at
        project["updated_at"] = now

        with path.open("w", encoding="utf-8") as handle:
            json.dump(project, handle, ensure_ascii=True, indent=2)

        return project

    def delete_project(self, project_id: str) -> None:
        path = self._project_path(project_id)
        if path.exists():
            path.unlink()

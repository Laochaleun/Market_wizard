"""Project management router."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models import DemographicProfile
from app.services import ProjectStore

router = APIRouter()


class ProjectPayload(BaseModel):
    """Payload for creating or updating a project."""

    id: Optional[str] = None
    name: str = Field(..., min_length=1)
    product_description: str = ""
    target_audience: Optional[DemographicProfile] = None
    research: dict[str, Any] = Field(default_factory=dict)


@router.get("/projects")
async def list_projects():
    """List saved projects."""
    store = ProjectStore()
    return store.list_projects()


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Fetch full project data."""
    store = ProjectStore()
    try:
        return store.load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/projects")
async def create_project(payload: ProjectPayload):
    """Create a new project."""
    store = ProjectStore()
    return store.save_project(payload.model_dump(mode="json"))


@router.put("/projects/{project_id}")
async def update_project(project_id: str, payload: ProjectPayload):
    """Update an existing project."""
    store = ProjectStore()
    data = payload.model_dump(mode="json")
    data["id"] = project_id
    return store.save_project(data)


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project."""
    store = ProjectStore()
    try:
        store.load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    store.delete_project(project_id)
    return {"status": "deleted", "id": project_id}

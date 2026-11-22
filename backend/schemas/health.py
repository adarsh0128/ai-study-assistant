"""Pydantic models for the health endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for GET /health.

    `status` is typed Literal["ok"] rather than str on purpose: it documents
    the contract in Swagger and makes it impossible for this endpoint to
    accidentally report any other value.
    """

    status: Literal["ok"] = Field(description="Always 'ok' when the server responds.")
    service: str = Field(description="Service identifier.")
    version: str = Field(description="API version from backend.__version__.")

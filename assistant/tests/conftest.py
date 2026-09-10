"""Make assistant's existing project-relative fixtures runnable from the workspace."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def assistant_project_directory(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

"""Regression tests for package metadata exposed to package indexes."""

from pathlib import Path
import tomllib


def _load_pyproject() -> dict[str, object]:
    """Load the project's ``pyproject.toml`` data."""
    project_root = Path(__file__).resolve().parents[1]
    return tomllib.loads((project_root / "pyproject.toml").read_text())


def test_project_declares_mit_license_and_source_url() -> None:
    """Ensure package indexes receive the license and source repository metadata."""
    project_data = _load_pyproject()["project"]

    assert project_data["license"] == "MIT"
    assert project_data["urls"]["Source"] == "https://github.com/langelabs/ai-track"


def test_readme_logo_uses_absolute_github_url() -> None:
    """Ensure the PyPI-rendered README can resolve the published logo asset."""
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text()

    assert "![ai-track logo](https://raw.githubusercontent.com/langelabs/ai-track/main/assets/logo_light.png)" in readme

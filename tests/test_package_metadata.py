"""Regression tests for package metadata exposed to package indexes."""

from pathlib import Path
from typing import Any
import tomllib


def _load_pyproject() -> dict[str, Any]:
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


def test_macos_extra_installs_mlx_audio_tts_dependencies() -> None:
    """Ensure the macOS extra installs the upstream MLX audio TTS dependency set."""
    project_data = _load_pyproject()["project"]
    macos_dependencies = project_data["optional-dependencies"]["macos"]

    assert any(dependency.startswith("mlx-audio[tts]>=0.4") for dependency in macos_dependencies)


def test_cuda_extra_requires_transformers_with_gemma4_support() -> None:
    """Ensure the CUDA extra installs a Transformers release that supports Gemma 4."""
    project_data = _load_pyproject()["project"]
    cuda_dependencies = project_data["optional-dependencies"]["cuda"]

    assert any(dependency.startswith("transformers>=5.5") for dependency in cuda_dependencies)

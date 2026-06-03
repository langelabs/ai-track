"""Safe diagnostics for local embedding backend loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any


def _safe_call(default: str, callback: object, *args: object) -> str:
    """Return a stringified callback result without allowing diagnostics to fail loading."""
    if not callable(callback):
        return default
    try:
        value = callback(*args)
    except Exception:
        return default
    return str(value)


def _safe_bool_call(callback: object) -> bool | None:
    """Return a boolean callback result, or ``None`` when unavailable."""
    if not callable(callback):
        return None
    try:
        return bool(callback())
    except Exception:
        return None


def _package_version(package_name: str) -> str:
    """Return an installed package version for diagnostics."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


@dataclass(frozen=True, slots=True)
class EmbeddingLoadDiagnostics:
    """Describe safe runtime state for embedding load diagnostics."""

    device: str
    torch_version: str
    torch_cuda_version: str
    transformers_version: str
    cuda_available: bool | None
    cuda_device: str
    cuda_memory_free: int | None
    cuda_memory_total: int | None

    def format(self) -> str:
        """Return a compact diagnostic string without secrets."""
        fields = [
            f"device={self.device}",
            f"torch={self.torch_version}",
            f"torch_cuda={self.torch_cuda_version}",
            f"transformers={self.transformers_version}",
            f"cuda_available={self.cuda_available}",
            f"cuda_device={self.cuda_device}",
            f"cuda_memory_free={self.cuda_memory_free}",
            f"cuda_memory_total={self.cuda_memory_total}",
        ]
        return " ".join(fields)


def collect_embedding_load_diagnostics(*, runtime: Any, device: str) -> EmbeddingLoadDiagnostics:
    """Collect safe embedding load diagnostics from the optional CUDA runtime."""
    torch = getattr(runtime, "torch", None)
    cuda = getattr(torch, "cuda", None)
    torch_version = str(getattr(torch, "__version__", "unavailable"))
    torch_cuda_version = str(getattr(getattr(torch, "version", None), "cuda", "unavailable"))
    cuda_available = _safe_bool_call(getattr(cuda, "is_available", None))
    cuda_device = "unavailable"
    cuda_memory_free: int | None = None
    cuda_memory_total: int | None = None
    if cuda_available:
        cuda_device = _safe_call("unavailable", getattr(cuda, "get_device_name", None), 0)
        mem_get_info = getattr(cuda, "mem_get_info", None)
        if callable(mem_get_info):
            try:
                free_memory, total_memory = mem_get_info()
            except Exception:
                pass
            else:
                cuda_memory_free = int(free_memory)
                cuda_memory_total = int(total_memory)
    return EmbeddingLoadDiagnostics(
        device=device,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        transformers_version=_package_version("transformers"),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        cuda_memory_free=cuda_memory_free,
        cuda_memory_total=cuda_memory_total,
    )


def build_embedding_load_error(
    *,
    model_id: str,
    phase: str,
    diagnostics: EmbeddingLoadDiagnostics,
    error: Exception,
) -> RuntimeError:
    """Return an actionable embedding load error for one failed phase."""
    return RuntimeError(
        f"CUDA embedding model load failed for {model_id} during {phase}: {error}. "
        f"{diagnostics.format()}"
    )

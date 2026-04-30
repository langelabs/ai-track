# ai-track

`ai-track` is a universal AI runtime library for local and remote inference.
It chooses the best available execution tier automatically, keeps the core
package lightweight, and exposes an OpenAI-style client surface so application
code can stay backend-agnostic.

## What it does

- Routes requests through local or remote inference automatically.
- Supports macOS MLX backends for on-device inference.
- Supports CUDA backends for GPU inference with vLLM and Hugging Face models.
- Falls back to a remote OpenAI-compatible client when no local backend fits.
- Exposes a familiar client surface for chat, embeddings, images, audio, and
  transcription.

## Architecture

The codebase is split into two major layers:

- `track.inference` contains the runtime primitives and backend implementations.
- `track.hub` contains the public routing layer that decides whether a model
  should use local inference or a remote client.

The runtime is centered around `LocalAI`, which can manage:

- chat generation
- embeddings
- image generation
- text-to-speech
- speech-to-text transcription

### Runtime selection

The runtime chooses a backend automatically when you do not pass one explicitly:

- macOS resolves to the MLX backend
- CUDA-capable Linux systems resolve to the CUDA backend
- everything else stays available through the remote OpenAI-compatible path

You can still force a backend explicitly when you need to.

## Local-first routing

Routing is local-first:

1. The hub checks whether the selected model is local.
2. If the runtime can serve it locally, the request stays on-device.
3. Otherwise the hub falls back to a remote OpenAI-compatible client.

This keeps local inference fast and private when available while preserving a
reliable remote fallback.

## Public API

The main entrypoints are:

```python
from track.hub import Hub
from track.inference import LocalAI
```

`LocalAI` exposes the local runtime directly and can also return an
OpenAI-style client.

`Hub` resolves a final client for a selected model and is the preferred way to
route requests from application code.

## OpenAI-style client

The local compatibility layer mirrors the shape of the OpenAI Python client.
It supports:

- `client.chat.completions.create(...)`
- `client.embeddings.create(...)`
- `client.images.generate(...)`
- `client.audio.speech.create(...)`
- `client.audio.transcriptions.create(...)`

### Example: chat

```python
from track.hub import Hub
from track.inference import AiModel, InferenceConfig, LocalAI

chat_model = AiModel(
    default=True,
    location="local",
    type="llm",
    status="available",
    model="mlx-community/qwen2",
    alias="Qwen2",
    inference_config=InferenceConfig(max_tokens=256, temperature=0.2),
)

runtime = LocalAI(
    chat_config=chat_model,
    remote_api_key="sk-example",
    remote_base_url="https://openrouter.ai/api/v1",
)

hub = Hub(local_ai=runtime)
client = hub.get_client(chat_model)

response = client.chat.completions.create(
    model=chat_model.model,
    messages=[
        {"role": "user", "content": "Summarize this architecture."},
    ],
)

print(response.choices[0].message["content"])
```

### Example: transcription

```python
from track.inference import LocalAI, TranscriptionModelConfig

runtime = LocalAI(
    backend="cuda",
    transcription_config=TranscriptionModelConfig(
        model_id="openai/whisper-small",
        alias="Whisper Small",
    ),
)

result = runtime.transcribe("sample.wav")
print(result.text)
```

### Example: OpenAI-style transcription

```python
client = runtime.get_client()
result = client.audio.transcriptions.create(
    model="openai/whisper-small",
    file="sample.wav",
)
print(result.text)
```

## Installation

The core package is intentionally small and works without the optional local
backends.

### Core install

```bash
uv sync
```

### macOS MLX extras

```bash
uv sync --extra macos
```

### CUDA extras

```bash
uv sync --extra cuda
```

The CUDA extra brings in the GPU-oriented runtime stack, including vLLM,
Transformers, Diffusers, and PyTorch-based helpers.

## Testing

Run the full unit suite with:

```bash
uv run python -m unittest discover -s tests
```

The tests focus on:

- hub routing decisions
- backend selection
- OpenAI-style client compatibility
- multimodal cleanup behavior
- transcription support
- CUDA factory selection

## Development notes

- Prefer `track.hub` for routing decisions.
- Keep optional imports lazy so the core package stays importable without MLX
  or CUDA dependencies.
- Add docstrings and type hints to new helpers and edited functions.
- Reuse shared helpers where both MLX and CUDA backends need the same logic.

"""FastAPI router factory for OpenAI-compatible hub endpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import APIRouter

    from track.hub import AiHub


def _serialize_response(value: Any) -> Any:
    """Convert OpenAI-compatible client values into JSON-safe objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _serialize_response(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_response(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_response(item) for item in value]
    return value


def _serialize_sse_event(value: Any) -> str:
    """Serialize one stream event as a server-sent event frame."""
    payload = json.dumps(_serialize_response(value), separators=(",", ":"))
    return f"data: {payload}\n\n"


def _stream_sse_events(events: Iterable[Any]) -> Iterator[str]:
    """Yield OpenAI-style server-sent events ending with the done marker."""
    for event in events:
        yield _serialize_sse_event(event)
    yield "data: [DONE]\n\n"


def _model_not_found_response(model_id: str) -> dict[str, Any]:
    """Build an OpenAI-style missing-model error payload."""
    return {
        "error": {
            "message": f"Model not found: {model_id}",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def _error_response(message: str, *, error_type: str, param: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style error response payload."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }


def _extract_model_id(payload: dict[str, Any]) -> str:
    """Return the request model id or raise a typed validation error."""
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Request body field 'model' must be a non-empty string.")
    return model_id


def _get_client(hub: AiHub, model_id: str) -> Any:
    """Return the client for a registered model or raise a missing-model error."""
    try:
        return hub.get_client(model_id)
    except KeyError as error:
        raise LookupError(model_id) from error


def _model_to_openai_payload(model: Any) -> dict[str, Any]:
    """Serialize one registered model in an OpenAI-compatible list shape."""
    return {
        "id": model.model_id,
        "object": "model",
        "created": 0,
        "owned_by": model.provider,
        "alias": model.alias,
    }


def _import_fastapi() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import FastAPI dependencies only when the API router is requested."""
    try:
        from fastapi import APIRouter, File, Form, Request, UploadFile
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as error:
        raise ImportError("FastAPI support requires installing ai-track with the 'api' extra.") from error
    globals().update({"Request": Request, "UploadFile": UploadFile})
    return APIRouter, File, Form, Request, UploadFile, JSONResponse, Response, StreamingResponse


def create_api_router(hub: AiHub) -> APIRouter:
    """Create a FastAPI router exposing OpenAI-compatible hub endpoints."""
    APIRouter, File, Form, Request, UploadFile, JSONResponse, Response, StreamingResponse = _import_fastapi()
    router = APIRouter()

    def handle_error(error: Exception) -> Any:
        """Translate common client errors into OpenAI-style HTTP responses."""
        if isinstance(error, LookupError):
            return JSONResponse(_model_not_found_response(str(error)), status_code=404)
        if isinstance(error, (TypeError, ValueError)):
            return JSONResponse(_error_response(str(error), error_type="invalid_request_error"), status_code=400)
        if isinstance(error, RuntimeError):
            return JSONResponse(_error_response(str(error), error_type="server_error"), status_code=500)
        raise error

    @router.get("/v1/models")
    async def list_models() -> Any:
        """List registered models in an OpenAI-compatible response shape."""
        return {"object": "list", "data": [_model_to_openai_payload(model) for model in hub.models]}

    @router.post("/v1/chat/completions")
    async def create_chat_completion(request: Request) -> Any:
        """Create or stream an OpenAI-compatible chat completion."""
        try:
            payload = await request.json()
            model_id = _extract_model_id(payload)
            result = _get_client(hub, model_id).chat.completions.create(**payload)
            if payload.get("stream") is True:
                return StreamingResponse(_stream_sse_events(result), media_type="text/event-stream")
            return JSONResponse(_serialize_response(result))
        except Exception as error:
            return handle_error(error)

    @router.post("/v1/embeddings")
    async def create_embedding(request: Request) -> Any:
        """Create OpenAI-compatible embeddings."""
        try:
            payload = await request.json()
            model_id = _extract_model_id(payload)
            result = _get_client(hub, model_id).embeddings.create(**payload)
            return JSONResponse(_serialize_response(result))
        except Exception as error:
            return handle_error(error)

    @router.post("/v1/images/generations")
    async def generate_image(request: Request) -> Any:
        """Generate or stream OpenAI-compatible images."""
        try:
            payload = await request.json()
            model_id = _extract_model_id(payload)
            result = _get_client(hub, model_id).images.generate(**payload)
            if payload.get("stream") is True:
                return StreamingResponse(_stream_sse_events(result), media_type="text/event-stream")
            return JSONResponse(_serialize_response(result))
        except Exception as error:
            return handle_error(error)

    @router.post("/v1/audio/speech")
    async def create_speech(request: Request) -> Any:
        """Generate speech audio from text input."""
        try:
            payload = await request.json()
            model_id = _extract_model_id(payload)
            result = _get_client(hub, model_id).audio.speech.create(**payload)
            return Response(content=result.audio, media_type=result.mime_type)
        except Exception as error:
            return handle_error(error)

    @router.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(default=None),
    ) -> Any:
        """Transcribe a multipart audio upload."""
        try:
            if not model.strip():
                raise ValueError("Form field 'model' must be a non-empty string.")
            contents = await file.read()
            result = _get_client(hub, model).audio.transcriptions.create(
                model=model,
                file=contents,
                language=language,
            )
            return JSONResponse(_serialize_response(result))
        except Exception as error:
            return handle_error(error)

    return router

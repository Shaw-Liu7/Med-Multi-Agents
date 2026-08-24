"""OpenAI-compatible LLM client with explicit, secret-safe configuration.

Configuration precedence is:

1. the ``config`` mapping passed to :class:`LLMClient`;
2. ``MEDIX_LLM_*`` environment variables;
3. a literal ``LLM_CONFIG`` mapping in ``config.py`` at the project root or
   its parent (legacy compatibility).

The fallback file is parsed with :mod:`ast`; it is never imported or executed.
This module also delays importing the optional ``openai`` package until a real
client is constructed, so configuration and protocol code remain testable
offline.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class LLMConfigurationError(ValueError):
    """Raised when an LLM client cannot be configured safely."""


class LLMProtocolError(RuntimeError):
    """Raised when an upstream response is not valid OpenAI-compatible data."""


@dataclass(frozen=True)
class ToolCall:
    """Normalized function-call data."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Normalized chat response with optional function calls."""

    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


_ENV_KEYS = {
    "api_key": "MEDIX_LLM_API_KEY",
    "base_url": "MEDIX_LLM_BASE_URL",
    "model_name": "MEDIX_LLM_MODEL",
    "temperature": "MEDIX_LLM_TEMPERATURE",
    "max_tokens": "MEDIX_LLM_MAX_TOKENS",
}

_PLACEHOLDERS = {
    "",
    "your-api-key",
    "your-llm-api-key",
    "your-your-model-api-key",
    "replace-me",
    "changeme",
}


def _read_literal_llm_config(path: Path) -> Dict[str, Any]:
    """Read a literal ``LLM_CONFIG`` assignment without executing the file."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise LLMConfigurationError(f"Cannot read LLM config file: {path.name}") from exc

    for node in tree.body:
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value

        if value is None or not any(
            isinstance(target, ast.Name) and target.id == "LLM_CONFIG"
            for target in targets
        ):
            continue

        try:
            parsed = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise LLMConfigurationError(
                "LLM_CONFIG must be a literal dictionary; executable expressions are forbidden"
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMConfigurationError("LLM_CONFIG must be a dictionary")
        return dict(parsed)

    return {}


def _default_config_candidates() -> List[Path]:
    project_root = Path(__file__).resolve().parents[1]
    return [project_root / "config.py", project_root.parent / "config.py"]


def _resolve_config_file(config_file: Optional[str | Path]) -> Optional[Path]:
    """Resolve a trusted relative config file location.

    Explicit paths must be relative to the project root and may not escape it.
    Absolute or ``..`` paths are rejected to avoid silently importing config
    from an unrelated machine location.
    """

    if config_file is None:
        return next((path for path in _default_config_candidates() if path.is_file()), None)

    raw = Path(config_file)
    if raw.is_absolute() or ".." in raw.parts:
        raise LLMConfigurationError("config_file must be a project-relative path")

    project_root = Path(__file__).resolve().parents[1]
    resolved = (project_root / raw).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise LLMConfigurationError("config_file escapes the project root") from exc
    if not resolved.is_file():
        raise LLMConfigurationError(f"Config file does not exist: {raw}")
    return resolved


def load_llm_config(
    explicit: Optional[Mapping[str, Any]] = None,
    *,
    config_file: Optional[str | Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Build an LLM configuration without logging or exposing credentials."""

    env = os.environ if environ is None else environ
    env_values = {
        key: env[env_name]
        for key, env_name in _ENV_KEYS.items()
        if env.get(env_name) not in (None, "")
    }

    # Passing an explicit mapping is an intentional self-contained
    # configuration boundary. Do not inspect a legacy file in that case. For
    # environment-only startup, consult the legacy file only when a required
    # live-client field is still absent. An explicitly named config_file is
    # always parsed because the caller requested it.
    should_load_implicit_file = (
        config_file is None
        and explicit is None
        and not (
            (env_values.get("model_name") or env_values.get("model"))
            and env_values.get("api_key")
        )
    )
    if config_file is not None:
        file_path = _resolve_config_file(config_file)
    elif should_load_implicit_file:
        file_path = _resolve_config_file(None)
    else:
        file_path = None
    merged: Dict[str, Any] = _read_literal_llm_config(file_path) if file_path else {}

    merged.update(env_values)

    # Explicit values win, including zero-valued temperature.
    if explicit:
        merged.update({key: value for key, value in explicit.items() if value is not None})

    # Accept the conventional ``model`` spelling while keeping one internal key.
    if "model_name" not in merged and "model" in merged:
        merged["model_name"] = merged["model"]

    if "temperature" in merged:
        try:
            merged["temperature"] = float(merged["temperature"])
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError("temperature must be a number") from exc
    else:
        merged["temperature"] = 0.2

    if "max_tokens" in merged:
        try:
            merged["max_tokens"] = int(merged["max_tokens"])
        except (TypeError, ValueError) as exc:
            raise LLMConfigurationError("max_tokens must be an integer") from exc
    else:
        merged["max_tokens"] = 4096

    return merged


def _validate_config(config: Mapping[str, Any], *, require_api_key: bool) -> None:
    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        raise LLMConfigurationError(
            "Missing model name; set MEDIX_LLM_MODEL or pass config['model_name']"
        )

    api_key = str(config.get("api_key", "")).strip()
    if require_api_key and api_key.lower() in _PLACEHOLDERS:
        raise LLMConfigurationError(
            "Missing LLM API key; set MEDIX_LLM_API_KEY or pass config['api_key']"
        )

    base_url = str(config.get("base_url", "https://api.openai.com/v1")).strip()
    parsed = urlparse(base_url)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LLMConfigurationError("base_url must be an absolute HTTP(S) URL")
    if parsed.scheme != "https" and parsed.hostname not in local_hosts:
        raise LLMConfigurationError("base_url must use HTTPS unless it targets localhost")

    temperature = config.get("temperature", 0.2)
    if not 0 <= float(temperature) <= 2:
        raise LLMConfigurationError("temperature must be between 0 and 2")
    if int(config.get("max_tokens", 4096)) <= 0:
        raise LLMConfigurationError("max_tokens must be positive")


class LLMClient:
    """Small adapter around an OpenAI-compatible async client.

    ``client`` is injectable so unit tests and private deployments can supply a
    compatible object without importing the OpenAI SDK.
    """

    def __init__(
        self,
        model_type: str = "openai_compatible",
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_file: Optional[str | Path] = None,
        client: Optional[Any] = None,
    ):
        if model_type != "openai_compatible":
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type
        self.config = load_llm_config(config, config_file=config_file)
        self.config.setdefault("base_url", "https://api.openai.com/v1")
        _validate_config(self.config, require_api_key=client is None)

        self.model_name = str(self.config["model_name"])
        self.temperature = float(self.config["temperature"])
        self.max_tokens = int(self.config["max_tokens"])

        if client is not None:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The optional 'openai' package is required for live LLM calls"
                ) from exc
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
            )

    @staticmethod
    def _request_value(value: Optional[Any], default: Any) -> Any:
        return default if value is None else value

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self._request_value(temperature, self.temperature),
            max_tokens=self._request_value(max_tokens, self.max_tokens),
            **kwargs,
        )
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise LLMProtocolError("LLM response did not contain a message choice") from exc

    async def chat_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3,
        **kwargs: Any,
    ) -> str:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        for attempt in range(max_retries):
            try:
                return await self.chat(messages, **kwargs)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                # Never log request bodies or exception strings: providers can
                # include headers or sensitive user text in them.
                logger.warning("LLM request failed; retrying (attempt %d)", attempt + 1)
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    @staticmethod
    def create_message(role: str, content: str) -> Dict[str, str]:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role}")
        return {"role": role, "content": content}

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if tool_choice not in {"auto", "required", "none"}:
            raise ValueError("tool_choice must be 'auto', 'required', or 'none'")

        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self._request_value(temperature, self.temperature),
            "max_tokens": self._request_value(max_tokens, self.max_tokens),
            **kwargs,
        }
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**request_params)
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as exc:
            raise LLMProtocolError("LLM response did not contain a message choice") from exc

        tool_calls: List[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("arguments are not an object")
                tool_calls.append(
                    ToolCall(
                        id=str(raw_call.id),
                        name=str(raw_call.function.name),
                        arguments=arguments,
                    )
                )
            except (AttributeError, TypeError, json.JSONDecodeError) as exc:
                raise LLMProtocolError("LLM returned a malformed tool call") from exc

        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            finish_reason=str(getattr(choice, "finish_reason", "unknown")),
        )

    @staticmethod
    def create_tool_message(
        tool_call_id: str,
        tool_name: str,
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        }


__all__ = [
    "LLMClient",
    "LLMConfigurationError",
    "LLMProtocolError",
    "LLMResponse",
    "ToolCall",
    "load_llm_config",
]

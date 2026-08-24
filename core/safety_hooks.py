"""Small compatibility layer for optional medical safety hooks.

The safety package is intentionally optional in the Alpha codebase. These
helpers tolerate older validators while allowing the newer ``SafetyGate`` API
to be enforced at both worker and coordinator boundaries.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional
from loguru import logger


DEFAULT_BLOCK_RESPONSE = (
    "抱歉，我不能安全地按当前方式处理这个请求。"
    "如果您正出现胸痛、呼吸困难、意识异常、大量出血或其他紧急症状，"
    "请立即联系当地急救服务或前往急诊。"
)


async def _invoke(
    method: Any,
    *,
    agent_id: str,
    question: str,
    output: Optional[str] = None,
    request_context: Any = None,
) -> Any:
    """Call a hook using its declared parameter names when possible."""
    if not callable(method):
        return None

    value_by_name = {
        "agent_id": agent_id,
        "agent": agent_id,
        "question": question,
        "query": question,
        "prompt": question,
        "user_input": question,
        "input_text": question,
        "input_data": question,
        "text": question if output is None else output,
        "request_context": request_context,
        "context": request_context,
        "output": output,
        "answer": output,
        "response": output,
        "final_output": output,
    }
    try:
        signature = inspect.signature(method)
        kwargs: Dict[str, Any] = {}
        unknown_required = []
        for name, parameter in signature.parameters.items():
            if name in ("self", "cls"):
                continue
            if name in value_by_name and value_by_name[name] is not None:
                kwargs[name] = value_by_name[name]
            elif parameter.default is inspect.Parameter.empty and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                unknown_required.append(name)
        if not unknown_required:
            result = method(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
    except (TypeError, ValueError):
        pass

    # Compatibility with early positional prototypes.
    attempts = []
    if output is None:
        attempts = [(agent_id, question), (question,)]
    else:
        attempts = [
            (agent_id, question, output),
            (agent_id, output),
            (question, output),
            (output,),
        ]
    last_error: Optional[Exception] = None
    for args in attempts:
        try:
            result = method(*args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except TypeError as exc:
            last_error = exc
    if last_error:
        logger.debug(f"Optional safety hook signature not recognized: {last_error}")
    return None


def _value(result: Any, *names: str) -> Any:
    if isinstance(result, Mapping):
        for name in names:
            if name in result:
                return result[name]
    for name in names:
        if hasattr(result, name):
            return getattr(result, name)
    return None


def _is_blocked(result: Any) -> bool:
    if result is None:
        return False
    blocked = _value(result, "blocked", "should_block", "deny")
    if blocked is not None:
        return bool(blocked)
    allowed = _value(result, "allowed", "safe", "valid")
    if allowed is not None:
        return not bool(allowed)
    action = str(_value(result, "action", "decision", "status") or "").lower()
    return action in {"block", "blocked", "deny", "denied", "reject", "rejected"}


def _response_text(result: Any) -> Optional[str]:
    if isinstance(result, str):
        return result
    value = _value(
        result,
        "safe_response",
        "replacement",
        "final_output",
        "output",
        "answer",
        "response",
        "message",
    )
    return str(value) if value else None


async def precheck_input(
    *,
    validator: Any,
    safety_gate: Any,
    agent_id: str,
    question: str,
    request_context: Any = None,
) -> Optional[str]:
    """Return a safe response when input must be blocked, otherwise ``None``."""
    hooks = (
        getattr(validator, "precheck_input", None),
        getattr(safety_gate, "check_input", None),
        getattr(safety_gate, "precheck_input", None),
    )
    for hook in hooks:
        if not callable(hook):
            continue
        try:
            result = await _invoke(
                hook,
                agent_id=agent_id,
                question=question,
                request_context=request_context,
            )
            if _is_blocked(result):
                return _response_text(result) or DEFAULT_BLOCK_RESPONSE
        except Exception as exc:
            logger.error(f"Input safety hook failed: error_type={type(exc).__name__}")
            # A configured medical safety component is a hard boundary.
            return DEFAULT_BLOCK_RESPONSE
    return None


async def finalize_output(
    *,
    validator: Any,
    safety_gate: Any,
    agent_id: str,
    question: str,
    output: str,
    request_context: Any = None,
) -> str:
    """Run hard output gates and return the approved/replaced response."""
    current = output or ""
    hooks = (
        getattr(validator, "gate_output", None),
        getattr(safety_gate, "finalize_output", None),
    )
    for hook in hooks:
        if not callable(hook):
            continue
        try:
            result = await _invoke(
                hook,
                agent_id=agent_id,
                question=question,
                output=current,
                request_context=request_context,
            )
            replacement = _response_text(result)
            if _is_blocked(result):
                current = replacement or DEFAULT_BLOCK_RESPONSE
            elif replacement:
                current = replacement
        except Exception as exc:
            logger.error(f"Output safety hook failed: error_type={type(exc).__name__}")
            current = DEFAULT_BLOCK_RESPONSE
    return current


__all__ = ["precheck_input", "finalize_output", "DEFAULT_BLOCK_RESPONSE"]

"""Immutable request context shared by routing, workers and synthesis.

Only canonical user/assistant messages belong to the conversation transcript.
Prompt text and worker tool traces are deliberately kept outside this model so
that a formatted prompt can never be recursively stored as conversation memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import re
import uuid


_IDENTIFIER_PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[身份证号已脱敏]"),
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[邮箱已脱敏]",
    ),
)
_SENSITIVE_KEYS = {"name", "full_name", "patient_name", "姓名"}
_SENSITIVE_KEY_PARTS = {
    "phone", "mobile", "telephone", "手机号", "电话",
    "id_card", "identity", "身份证", "证件号",
    "email", "邮箱", "address", "住址", "地址",
    "wechat", "微信", "qq", "mrn", "medical_record", "病历号",
    "dob", "date_of_birth", "出生日期",
}


def redact_text(value: Any) -> Tuple[str, bool]:
    """Deterministically remove common direct identifiers."""
    text = str(value or "")
    changed = False
    for pattern, replacement in _IDENTIFIER_PATTERNS:
        text, count = pattern.subn(replacement, text)
        changed = changed or count > 0
    return text, changed


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in _SENSITIVE_KEYS or any(
        part in normalized for part in _SENSITIVE_KEY_PARTS
    )


def redact_sensitive_data(value: Any) -> Tuple[Any, bool]:
    """Recursively redact strings and replace sensitive-key values."""
    if isinstance(value, Mapping):
        output = {}
        changed = False
        for key, item in value.items():
            if _is_sensitive_key(key):
                output[str(key)] = "[敏感字段已脱敏]"
                changed = True
            else:
                redacted, item_changed = redact_sensitive_data(item)
                output[str(key)] = redacted
                changed = changed or item_changed
        return output, changed
    if isinstance(value, (list, tuple, set, frozenset)):
        output = []
        changed = False
        for item in value:
            redacted, item_changed = redact_sensitive_data(item)
            output.append(redacted)
            changed = changed or item_changed
        return tuple(output), changed
    if isinstance(value, str):
        return redact_text(value)
    return value, False


def _freeze(value: Any) -> Any:
    """Recursively convert mutable containers to immutable snapshots."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    return value


def thaw(value: Any) -> Any:
    """Return a JSON-friendly copy of a frozen value."""
    if isinstance(value, Mapping):
        return {str(key): thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def _strict_bool(value: Any) -> bool:
    """Parse consent without treating arbitrary non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    # Consent is fail-closed for unknown representations.
    return False


@dataclass(frozen=True)
class ConversationTurn:
    """One complete canonical user/assistant exchange."""

    user: str
    assistant: str
    turn_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        user, _ = redact_text(self.user)
        assistant, _ = redact_text(self.assistant)
        object.__setattr__(self, "user", user.strip())
        object.__setattr__(self, "assistant", assistant.strip())


@dataclass(frozen=True)
class RetrievedMemory:
    """A bounded, attributable long-term-memory search result."""

    content: str
    score: float = 0.0
    memory_id: str = ""
    source: str = "long_term_memory"
    timestamp: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        content, _ = redact_text(self.content)
        metadata, _ = redact_sensitive_data(dict(self.metadata or {}))
        object.__setattr__(self, "content", content.strip())
        try:
            score = float(self.score)
        except (TypeError, ValueError):
            score = 0.0
        object.__setattr__(self, "score", score if isfinite(score) else 0.0)
        object.__setattr__(self, "metadata", _freeze(metadata))


def complete_turns_from_messages(
    messages: Iterable[Any],
    *,
    limit: int = 5,
) -> Tuple[ConversationTurn, ...]:
    """Build recent complete turns, dropping tools and orphaned messages.

    A new user message replaces an unfinished user message. This prevents a
    truncated history window from starting with an assistant response.
    """
    completed = []
    pending_user: Optional[Dict[str, str]] = None

    for message in messages or ():
        if isinstance(message, ConversationTurn):
            if message.user and message.assistant:
                completed.append(message)
            pending_user = None
            continue

        if isinstance(message, Mapping):
            role = str(message.get("role", "")).lower()
            content = str(message.get("content", "")).strip()
            timestamp = str(message.get("timestamp", ""))
            message_turn_id = str(message.get("turn_id", ""))
        else:
            role = str(getattr(message, "role", "")).lower()
            content = str(getattr(message, "content", "")).strip()
            timestamp = str(getattr(message, "timestamp", ""))
            message_turn_id = str(getattr(message, "turn_id", ""))

        if not content:
            continue
        if role == "user":
            pending_user = {
                "content": content,
                "timestamp": timestamp,
                "turn_id": message_turn_id,
            }
        elif role == "assistant" and pending_user is not None:
            completed.append(ConversationTurn(
                user=pending_user["content"],
                assistant=content,
                turn_id=message_turn_id or pending_user["turn_id"],
                timestamp=pending_user["timestamp"] or timestamp,
            ))
            pending_user = None

    if limit <= 0:
        return ()
    return tuple(completed[-limit:])


def memories_from_results(
    results: Iterable[Any],
    *,
    limit: int = 3,
    min_score: float = 0.55,
) -> Tuple[RetrievedMemory, ...]:
    """Normalize and bound memory results before they enter a prompt."""
    normalized = []
    seen = set()
    for item in results or ():
        if isinstance(item, RetrievedMemory):
            memory = item
        elif isinstance(item, Mapping):
            memory = RetrievedMemory(
                memory_id=str(item.get("memory_id", item.get("id", ""))),
                content=str(item.get("content", item.get("memory", item.get("summary", "")))),
                score=item.get("score", 0.0),
                source=str(item.get("source", "long_term_memory")),
                timestamp=str(item.get("timestamp", "")),
                metadata=item.get("metadata", {}) or {},
            )
        else:
            continue

        fingerprint = " ".join(memory.content.lower().split())
        if not fingerprint or fingerprint in seen or memory.score < min_score:
            continue
        seen.add(fingerprint)
        normalized.append(memory)

    normalized.sort(key=lambda item: item.score, reverse=True)
    return tuple(normalized[:max(0, limit)])


@dataclass(frozen=True)
class ConversationContext:
    """Immutable snapshot for exactly one user turn."""

    tenant_id: str
    user_id: str
    session_id: str
    turn_id: str
    raw_question: str
    patient_profile: Mapping[str, Any] = field(default_factory=dict)
    rolling_summary: str = ""
    recent_turns: Tuple[ConversationTurn, ...] = ()
    retrieved_memories: Tuple[RetrievedMemory, ...] = ()
    task_instruction: str = ""
    collaboration_results: Tuple[Mapping[str, Any], ...] = ()
    long_term_memory_consent: bool = False
    input_redacted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "user_id", "session_id", "turn_id"):
            value = str(getattr(self, field_name, "")).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)

        question, question_redacted = redact_text(self.raw_question)
        question = question.strip()
        if not question:
            raise ValueError("raw_question must not be empty")
        object.__setattr__(self, "raw_question", question)
        rolling_summary, summary_redacted = redact_text(self.rolling_summary)
        task_instruction, task_redacted = redact_text(self.task_instruction)
        patient_profile, profile_redacted = redact_sensitive_data(
            dict(self.patient_profile or {})
        )
        collaboration, collaboration_redacted = redact_sensitive_data(
            tuple(self.collaboration_results or ())
        )
        metadata, metadata_redacted = redact_sensitive_data(dict(self.metadata or {}))
        object.__setattr__(self, "rolling_summary", rolling_summary.strip())
        object.__setattr__(self, "task_instruction", task_instruction.strip())
        object.__setattr__(
            self,
            "long_term_memory_consent",
            _strict_bool(self.long_term_memory_consent),
        )
        object.__setattr__(self, "patient_profile", _freeze(patient_profile))
        object.__setattr__(self, "recent_turns", tuple(self.recent_turns or ()))
        object.__setattr__(self, "retrieved_memories", tuple(self.retrieved_memories or ()))
        object.__setattr__(
            self,
            "collaboration_results",
            tuple(
                _freeze(dict(item))
                for item in collaboration
                if isinstance(item, Mapping)
            ),
        )
        object.__setattr__(self, "metadata", _freeze(metadata))
        object.__setattr__(
            self,
            "input_redacted",
            bool(self.input_redacted) or any((
                question_redacted,
                summary_redacted,
                task_redacted,
                profile_redacted,
                collaboration_redacted,
                metadata_redacted,
            )),
        )

    @property
    def storage_session_id(self) -> str:
        """Collision-free key for legacy memory backends without user scoping."""
        return f"{self.tenant_id}:{self.user_id}:{self.session_id}"

    @property
    def memory_consent(self) -> bool:
        """Backward-compatible name for explicit long-term persistence consent."""
        return self.long_term_memory_consent

    def for_task(
        self,
        instruction: str,
        *,
        collaboration_results: Sequence[Mapping[str, Any]] = (),
    ) -> "ConversationContext":
        """Create an immutable task-specific view without mutating this turn."""
        return replace(
            self,
            task_instruction=instruction,
            collaboration_results=tuple(collaboration_results),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "raw_question": self.raw_question,
            "patient_profile": thaw(self.patient_profile),
            "rolling_summary": self.rolling_summary,
            "recent_turns": [
                {
                    "user": turn.user,
                    "assistant": turn.assistant,
                    "turn_id": turn.turn_id,
                    "timestamp": turn.timestamp,
                }
                for turn in self.recent_turns
            ],
            "retrieved_memories": [
                {
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "score": memory.score,
                    "source": memory.source,
                    "timestamp": memory.timestamp,
                    "metadata": thaw(memory.metadata),
                }
                for memory in self.retrieved_memories
            ],
            "task_instruction": self.task_instruction,
            "collaboration_results": [thaw(item) for item in self.collaboration_results],
            "long_term_memory_consent": self.long_term_memory_consent,
            "input_redacted": self.input_redacted,
            "metadata": thaw(self.metadata),
        }

    @classmethod
    def from_legacy(
        cls,
        raw_question: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> "ConversationContext":
        """Normalize the old ``question/context/session_id`` public API."""
        legacy = dict(context or {})
        resolved_session = str(session_id or legacy.pop("session_id", "") or uuid.uuid4())
        resolved_tenant = str(tenant_id or legacy.pop("tenant_id", "") or "default")

        # Anonymous callers are deliberately session-scoped. Cross-session
        # memory requires an explicit stable user_id from the application.
        resolved_user = str(
            user_id
            or legacy.pop("user_id", "")
            or f"anonymous:{resolved_session}"
        )
        resolved_turn = str(turn_id or legacy.pop("turn_id", "") or uuid.uuid4())

        profile = legacy.pop("patient_profile", None)
        recent = legacy.pop("recent_turns", legacy.pop("recent_history", ()))
        memories = legacy.pop("retrieved_memories", legacy.pop("historical_cases", ()))
        rolling_summary = str(legacy.pop("rolling_summary", legacy.pop("summary", "")) or "")
        memory_consent = _strict_bool(
            legacy.pop(
                "long_term_memory_consent",
                legacy.pop("memory_consent", legacy.pop("allow_long_term_memory", False)),
            )
        )
        metadata = legacy.pop("metadata", {}) or {}
        input_redacted = _strict_bool(legacy.pop("input_redacted", False))

        # Legacy context dictionaries commonly put age/history directly at the
        # top level. Preserve those values as structured patient state.
        if profile is None:
            profile = legacy
        elif legacy:
            profile = {**legacy, **dict(profile)}

        recent_values = tuple(recent or ())
        if recent_values and all(
            isinstance(item, ConversationTurn) for item in recent_values
        ):
            turns = recent_values[-5:]
        else:
            turns = complete_turns_from_messages(recent_values, limit=5)

        return cls(
            tenant_id=resolved_tenant,
            user_id=resolved_user,
            session_id=resolved_session,
            turn_id=resolved_turn,
            raw_question=raw_question,
            patient_profile=profile or {},
            rolling_summary=rolling_summary,
            recent_turns=turns,
            retrieved_memories=memories_from_results(memories, limit=3),
            long_term_memory_consent=memory_consent,
            input_redacted=input_redacted,
            metadata=metadata,
        )


# ``RequestContext`` is the public request-oriented name; both names denote the
# same immutable model to keep integrations simple.
RequestContext = ConversationContext


__all__ = [
    "ConversationContext",
    "RequestContext",
    "ConversationTurn",
    "RetrievedMemory",
    "complete_turns_from_messages",
    "memories_from_results",
    "redact_text",
    "redact_sensitive_data",
    "thaw",
]

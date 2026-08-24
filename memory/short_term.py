"""会话级短期记忆。

设计要点：
- 实例由上层显式创建和注入，不使用危险的进程全局单例。
- 以 tenant/user/session 三元组隔离数据。
- 工具调用轨迹与面向用户的 user/assistant 对话分开存储。
- 上下文按完整回合和字符预算裁剪，旧回合进入滚动摘录。
- Redis 追加使用 list + transaction，不再读取整个 JSON 后覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import asyncio
import hashlib
import json
import uuid

from loguru import logger

from .entropy_manager import MemoryEntropyManager


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any, default: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return default or _utcnow()


@dataclass(frozen=True)
class MemoryScope:
    """记忆隔离边界。"""

    tenant_id: str
    user_id: str
    session_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("tenant_id", self.tenant_id),
            ("user_id", self.user_id),
            ("session_id", self.session_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if len(value) > 512:
                raise ValueError(f"{name} is too long")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [self.tenant_id, self.user_id, self.session_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def cache_key(self) -> str:
        # 内存键不暴露原始用户标识，同时消除 ':' 等分隔符冲突。
        return self.digest


@dataclass
class ConversationHistory:
    """一个隔离作用域内的对话。"""

    # 前五个字段保持旧版位置参数顺序。
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    last_updated: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"
    user_id: str = "anonymous"
    traces: List[Dict[str, Any]] = field(default_factory=list)
    rolling_summary: str = ""
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @property
    def scope(self) -> MemoryScope:
        return MemoryScope(self.tenant_id, self.user_id, self.session_id)

    @staticmethod
    def _is_trace(role: str, content: str, message_type: Optional[str]) -> bool:
        if message_type in {"tool", "trace", "internal"}:
            return True
        if role in {"tool", "function"}:
            return True
        return role == "assistant" and content.lstrip().startswith("调用工具：")

    def add_message(
        self,
        role: str,
        content: str,
        *,
        message_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        content = str(content)
        is_trace = self._is_trace(role, content, message_type)
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": (timestamp or _utcnow()).isoformat(),
            "message_type": message_type or ("trace" if is_trace else "dialogue"),
        }
        if metadata:
            message["metadata"] = dict(metadata)
        with self._lock:
            (self.traces if is_trace else self.messages).append(message)
            self.last_updated = _utcnow()
        return message

    def add_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """原子追加一个完整回合。"""
        now = _utcnow()
        turn_id = str(uuid.uuid4())
        common_metadata = {"turn_id": turn_id, **dict(metadata or {})}
        user = {
            "message_id": str(uuid.uuid4()),
            "role": "user",
            "content": str(user_content),
            "timestamp": now.isoformat(),
            "message_type": "dialogue",
            "metadata": common_metadata,
        }
        assistant = {
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": str(assistant_content),
            "timestamp": _utcnow().isoformat(),
            "message_type": "dialogue",
            "metadata": common_metadata,
        }
        with self._lock:
            self.messages.extend((user, assistant))
            self.last_updated = _utcnow()
        return user, assistant

    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(message) for message in self.messages[-max(0, limit):]]

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 2,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "messages": list(self.messages),
                "traces": list(self.traces),
                "rolling_summary": self.rolling_summary,
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat(),
                "metadata": dict(self.metadata),
            }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConversationHistory":
        return cls(
            session_id=str(data["session_id"]),
            messages=[dict(item) for item in data.get("messages", [])],
            created_at=_parse_datetime(data.get("created_at")),
            last_updated=_parse_datetime(data.get("last_updated")),
            metadata=dict(data.get("metadata") or {}),
            tenant_id=str(data.get("tenant_id") or "default"),
            user_id=str(data.get("user_id") or "anonymous"),
            traces=[dict(item) for item in data.get("traces", [])],
            rolling_summary=str(data.get("rolling_summary") or ""),
        )


class ShortTermMemory:
    """可显式配置的短期记忆实例。

    这个类是同步的；Redis 客户端操作可能阻塞。异步服务可使用
    ``aadd_message``/``aget_history`` 包装到工作线程，或在应用层注入
    原生异步存储实现。
    """

    def __init__(
        self,
        storage_type: str = "memory",
        redis_config: Optional[Dict[str, Any]] = None,
        *,
        redis_client: Any = None,
        default_tenant_id: str = "default",
        default_user_id: str = "anonymous",
        redis_ttl_seconds: int = 3600,
        max_stored_messages: int = 200,
        max_trace_entries: int = 200,
        max_context_chars: int = 12000,
        summary_char_budget: int = 3000,
        entropy_manager: Optional[MemoryEntropyManager] = None,
        fallback_to_memory: bool = True,
    ) -> None:
        if storage_type not in {"memory", "redis"}:
            raise ValueError("storage_type must be 'memory' or 'redis'")
        if redis_ttl_seconds <= 0 or max_stored_messages < 2 or max_trace_entries < 1:
            raise ValueError("storage limits must be positive")
        if max_context_chars < 256 or summary_char_budget < 128:
            raise ValueError("context budgets are too small")

        # 校验默认作用域。
        MemoryScope(default_tenant_id, default_user_id, "validation")
        self.storage_type = storage_type
        self.default_tenant_id = default_tenant_id
        self.default_user_id = default_user_id
        self.redis_ttl_seconds = redis_ttl_seconds
        self.max_stored_messages = max_stored_messages
        self.max_trace_entries = max_trace_entries
        self.max_context_chars = max_context_chars
        self.summary_char_budget = summary_char_budget
        self.entropy_manager = entropy_manager or MemoryEntropyManager()
        self.sessions: Dict[str, ConversationHistory] = {}
        self.redis_client = redis_client
        self._lock = RLock()

        if storage_type == "redis" and self.redis_client is None:
            try:
                import redis  # type: ignore

                config = dict(redis_config or {})
                self.redis_client = redis.Redis(
                    host=config.get("host", "localhost"),
                    port=config.get("port", 6379),
                    db=config.get("db", 0),
                    password=config.get("password"),
                    ssl=bool(config.get("ssl", False)),
                    socket_timeout=config.get("socket_timeout", 2),
                    decode_responses=True,
                )
                self.redis_client.ping()
            except Exception as exc:
                if not fallback_to_memory:
                    raise RuntimeError("Redis short-term memory initialization failed") from exc
                logger.warning(
                    "Redis unavailable; using process-local memory: {}",
                    type(exc).__name__,
                )
                self.storage_type = "memory"
                self.redis_client = None

    def _scope(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> MemoryScope:
        return MemoryScope(
            tenant_id or self.default_tenant_id,
            user_id or self.default_user_id,
            session_id,
        )

    @staticmethod
    def _redis_keys(scope: MemoryScope) -> Dict[str, str]:
        prefix = f"medix:memory:v2:{scope.digest}"
        return {
            "meta": f"{prefix}:meta",
            "messages": f"{prefix}:messages",
            "traces": f"{prefix}:traces",
        }

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def create_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ConversationHistory:
        scope = self._scope(session_id, tenant_id, user_id)
        if self.storage_type == "memory":
            with self._lock:
                history = self.sessions.get(scope.cache_key)
                if history is None:
                    history = ConversationHistory(
                        session_id=scope.session_id,
                        metadata=dict(metadata or {}),
                        tenant_id=scope.tenant_id,
                        user_id=scope.user_id,
                    )
                    self.sessions[scope.cache_key] = history
                elif metadata:
                    history.metadata.update(metadata)
                    history.last_updated = _utcnow()
                return history

        self._ensure_redis_session(scope, metadata)
        return self.get_session(session_id, tenant_id=scope.tenant_id, user_id=scope.user_id) or ConversationHistory(
            session_id=scope.session_id,
            metadata=dict(metadata or {}),
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
        )

    def _ensure_redis_session(
        self,
        scope: MemoryScope,
        metadata: Optional[Mapping[str, Any]] = None,
        pipeline: Any = None,
    ) -> Any:
        if self.redis_client is None:
            return pipeline
        target = pipeline or self.redis_client.pipeline(transaction=True)
        keys = self._redis_keys(scope)
        now = _utcnow().isoformat()
        mapping = {
            "schema_version": "2",
            "tenant_id": scope.tenant_id,
            "user_id": scope.user_id,
            "session_id": scope.session_id,
            "last_updated": now,
        }
        # created_at 只在首次创建时设置。
        target.hsetnx(keys["meta"], "created_at", now)
        target.hsetnx(keys["meta"], "metadata", "{}")
        target.hset(keys["meta"], mapping=mapping)
        if metadata is not None:
            target.hset(keys["meta"], "metadata", self._json(dict(metadata)))
        for key in keys.values():
            target.expire(key, self.redis_ttl_seconds)
        if pipeline is None:
            target.execute()
        return target

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        message_type: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        scope = self._scope(session_id, tenant_id, user_id)
        is_trace = ConversationHistory._is_trace(role, str(content), message_type)
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": str(content),
            "timestamp": _utcnow().isoformat(),
            "message_type": message_type or ("trace" if is_trace else "dialogue"),
        }
        if metadata:
            message["metadata"] = dict(metadata)

        if self.storage_type == "memory":
            with self._lock:
                history = self.sessions.get(scope.cache_key)
                if history is None:
                    history = ConversationHistory(
                        session_id=scope.session_id,
                        tenant_id=scope.tenant_id,
                        user_id=scope.user_id,
                    )
                    self.sessions[scope.cache_key] = history
                with history._lock:
                    target = history.traces if is_trace else history.messages
                    target.append(message)
                    if is_trace and len(target) > self.max_trace_entries:
                        del target[:-self.max_trace_entries]
                    history.last_updated = _utcnow()
                    if not is_trace:
                        self._compact_history(history)
            return dict(message)

        if self.redis_client is None:
            raise RuntimeError("Redis storage selected without a Redis client")
        keys = self._redis_keys(scope)
        target_key = keys["traces"] if is_trace else keys["messages"]
        pipe = self.redis_client.pipeline(transaction=True)
        self._ensure_redis_session(scope, pipeline=pipe)
        pipe.rpush(target_key, self._json(message))
        if is_trace:
            pipe.ltrim(target_key, -self.max_trace_entries, -1)
        pipe.expire(target_key, self.redis_ttl_seconds)
        pipe.execute()
        if not is_trace:
            self._compact_redis_scope(scope)
        return dict(message)

    def add_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        scope = self._scope(session_id, tenant_id, user_id)
        turn_id = str(uuid.uuid4())
        turn_metadata = {"turn_id": turn_id, **dict(metadata or {})}
        now = _utcnow().isoformat()
        user = {
            "message_id": str(uuid.uuid4()), "role": "user", "content": str(user_content),
            "timestamp": now, "message_type": "dialogue", "metadata": turn_metadata,
        }
        assistant = {
            "message_id": str(uuid.uuid4()), "role": "assistant", "content": str(assistant_content),
            "timestamp": _utcnow().isoformat(), "message_type": "dialogue", "metadata": turn_metadata,
        }
        if self.storage_type == "memory":
            with self._lock:
                history = self.sessions.get(scope.cache_key)
                if history is None:
                    history = ConversationHistory(
                        session_id=scope.session_id,
                        tenant_id=scope.tenant_id,
                        user_id=scope.user_id,
                    )
                    self.sessions[scope.cache_key] = history
                with history._lock:
                    history.messages.extend((user, assistant))
                    history.last_updated = _utcnow()
                    self._compact_history(history)
        else:
            if self.redis_client is None:
                raise RuntimeError("Redis storage selected without a Redis client")
            keys = self._redis_keys(scope)
            pipe = self.redis_client.pipeline(transaction=True)
            self._ensure_redis_session(scope, pipeline=pipe)
            pipe.rpush(keys["messages"], self._json(user), self._json(assistant))
            pipe.expire(keys["messages"], self.redis_ttl_seconds)
            pipe.execute()
            self._compact_redis_scope(scope)
        return dict(user), dict(assistant)

    def get_session(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[ConversationHistory]:
        scope = self._scope(session_id, tenant_id, user_id)
        if self.storage_type == "memory":
            with self._lock:
                return self.sessions.get(scope.cache_key)
        return self._load_from_redis(scope)

    def _compact_history(self, history: ConversationHistory) -> bool:
        """将旧完整回合写入 rolling_summary，并真正从原消息中删除。"""
        original_messages = list(history.messages)
        # 为旧版/手工注入的消息补齐稳定 ID，后续才能精确删除已摘录回合。
        for message in original_messages:
            message.setdefault("message_id", str(uuid.uuid4()))
            message.setdefault("message_type", "dialogue")
        complete = self.entropy_manager.complete_turns(original_messages)
        chars = sum(len(str(message.get("content", ""))) for message in original_messages)
        storage_char_limit = self.max_context_chars * 2
        if len(original_messages) <= self.max_stored_messages and chars <= storage_char_limit:
            return False

        keep_turns = max(1, min(len(complete), self.max_stored_messages // 2))
        while keep_turns > 1:
            recent_chars = sum(
                len(str(message.get("content", "")))
                for turn in complete[-keep_turns:]
                for message in turn
            )
            if recent_chars <= storage_char_limit:
                break
            keep_turns -= 1
        older = complete[:-keep_turns] if complete else []
        if not older:
            # 单个超大回合不裁碎；上下文读取时再做字符预算副本。
            return False

        removed_ids = {
            str(message.get("message_id"))
            for turn in older
            for message in turn
            if message.get("message_id")
        }
        removed_objects = {id(message) for turn in older for message in turn}
        excerpt = self.entropy_manager._deterministic_summary(older, self.summary_char_budget)
        if excerpt:
            combined = "\n".join(part for part in (history.rolling_summary.strip(), excerpt) if part)
            history.rolling_summary = combined[-self.summary_char_budget:]
        history.messages = [
            message for message in original_messages
            if (
                (message.get("message_id") and str(message.get("message_id")) not in removed_ids)
                or (not message.get("message_id") and id(message) not in removed_objects)
            )
        ]
        history.last_updated = _utcnow()
        return history.messages != original_messages

    def _compact_redis_scope(self, scope: MemoryScope) -> None:
        """WATCH + transaction 避免压缩时覆盖其他进程的新追加。"""
        if self.redis_client is None:
            return
        keys = self._redis_keys(scope)
        for _ in range(3):
            pipe = self.redis_client.pipeline(transaction=True)
            try:
                pipe.watch(keys["messages"], keys["meta"])
                raw_messages = pipe.lrange(keys["messages"], 0, -1)
                meta = pipe.hgetall(keys["meta"]) or {}
                messages = [json.loads(item) for item in raw_messages]
                history = ConversationHistory(
                    session_id=scope.session_id,
                    messages=messages,
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    metadata=json.loads(meta.get("metadata", "{}")),
                    rolling_summary=meta.get("rolling_summary", ""),
                    created_at=_parse_datetime(meta.get("created_at")),
                    last_updated=_parse_datetime(meta.get("last_updated")),
                )
                if not self._compact_history(history):
                    pipe.unwatch()
                    return
                pipe.multi()
                pipe.delete(keys["messages"])
                if history.messages:
                    pipe.rpush(keys["messages"], *(self._json(item) for item in history.messages))
                pipe.hset(keys["meta"], mapping={
                    "rolling_summary": history.rolling_summary,
                    "last_updated": history.last_updated.isoformat(),
                })
                pipe.expire(keys["messages"], self.redis_ttl_seconds)
                pipe.expire(keys["meta"], self.redis_ttl_seconds)
                pipe.execute()
                return
            except Exception as exc:
                # redis.WatchError 在此重试；其他客户端异常在三次后记录。
                try:
                    pipe.reset()
                except Exception:
                    pass
                last_error = exc
        logger.warning(
            "Short-term Redis compaction skipped after concurrent changes: {}",
            type(last_error).__name__,
        )

    def _load_from_redis(self, scope: MemoryScope) -> Optional[ConversationHistory]:
        if self.redis_client is None:
            return None
        keys = self._redis_keys(scope)
        try:
            pipe = self.redis_client.pipeline(transaction=False)
            pipe.hgetall(keys["meta"])
            pipe.lrange(keys["messages"], 0, -1)
            pipe.lrange(keys["traces"], 0, -1)
            meta, raw_messages, raw_traces = pipe.execute()
            if not meta and not raw_messages and not raw_traces:
                return None
            return ConversationHistory(
                session_id=scope.session_id,
                messages=[json.loads(item) for item in raw_messages],
                traces=[json.loads(item) for item in raw_traces],
                created_at=_parse_datetime(meta.get("created_at")),
                last_updated=_parse_datetime(meta.get("last_updated")),
                metadata=json.loads(meta.get("metadata", "{}")),
                tenant_id=scope.tenant_id,
                user_id=scope.user_id,
                rolling_summary=meta.get("rolling_summary", ""),
            )
        except Exception as exc:
            logger.error("Failed to load short-term memory: {}", type(exc).__name__)
            return None

    @staticmethod
    def _clip_turn(
        user: Mapping[str, Any],
        assistant: Mapping[str, Any],
        budget: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """仅对传给模型的副本裁剪，存储原文不受影响。"""
        user_copy, assistant_copy = dict(user), dict(assistant)
        marker = "……[按上下文预算裁剪]"
        available = max(32, budget - len(marker) * 2)
        user_budget = max(16, available // 3)
        assistant_budget = max(16, available - user_budget)
        user_text = str(user_copy.get("content", ""))
        assistant_text = str(assistant_copy.get("content", ""))
        if len(user_text) > user_budget:
            user_copy["content"] = user_text[:user_budget] + marker
        if len(assistant_text) > assistant_budget:
            assistant_copy["content"] = assistant_text[:assistant_budget] + marker
        return user_copy, assistant_copy

    def _context_messages(
        self,
        history: ConversationHistory,
        *,
        message_limit: int,
        turn_limit: Optional[int],
        char_budget: int,
    ) -> List[Dict[str, Any]]:
        if message_limit <= 0 or char_budget <= 0:
            return []
        with history._lock:
            turns = self.entropy_manager.complete_turns(history.messages)
            summary = history.rolling_summary.strip()

        max_turns_by_count = message_limit // 2
        if summary and message_limit >= 3:
            max_turns_by_count = (message_limit - 1) // 2
        if turn_limit is not None:
            max_turns_by_count = min(max_turns_by_count, max(0, turn_limit))

        selected_reversed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        used = 0
        for user, assistant in reversed(turns[-max_turns_by_count:] if max_turns_by_count else []):
            pair_chars = len(str(user.get("content", ""))) + len(str(assistant.get("content", "")))
            remaining = char_budget - used
            if pair_chars <= remaining:
                selected_reversed.append((dict(user), dict(assistant)))
                used += pair_chars
            elif not selected_reversed and remaining >= 64:
                selected_reversed.append(self._clip_turn(user, assistant, remaining))
                used = char_budget
            else:
                break

        selected = list(reversed(selected_reversed))
        result: List[Dict[str, Any]] = []
        if summary and len(result) < message_limit:
            remaining = max(0, char_budget - used)
            if remaining >= 32:
                summary_text = summary[-min(len(summary), remaining):]
                result.append({
                    "role": "system",
                    "content": "[早期对话滚动摘录，非逐字原文]\n" + summary_text,
                    "message_type": "summary",
                })
        for user, assistant in selected:
            result.extend((user, assistant))
        return result[:message_limit]

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 50,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        char_budget: Optional[int] = None,
        turn_limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        scope = self._scope(session_id, tenant_id, user_id)
        if self.storage_type == "redis":
            self._compact_redis_scope(scope)
        history = self.get_session(
            session_id, tenant_id=scope.tenant_id, user_id=scope.user_id
        )
        if history is None:
            return []
        if self.storage_type == "memory":
            with self._lock, history._lock:
                self._compact_history(history)  # 清理直接写回内存对象。
        return self._context_messages(
            history,
            message_limit=limit,
            turn_limit=turn_limit,
            char_budget=char_budget or self.max_context_chars,
        )

    def get_history(
        self,
        session_id: str,
        limit: int = 10,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        char_budget: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """返回最多 ``limit`` 个完整回合，可选带一条滚动摘要。"""
        if limit <= 0:
            return []
        messages = self.get_recent_messages(
            session_id,
            limit=max(1, limit * 2 + 1),
            tenant_id=tenant_id,
            user_id=user_id,
            char_budget=char_budget,
            turn_limit=limit,
        )
        return [
            {"role": str(message["role"]), "content": str(message.get("content", ""))}
            for message in messages
            if message.get("role") in {"system", "user", "assistant"}
        ]

    def get_tool_traces(
        self,
        session_id: str,
        limit: int = 50,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        history = self.get_session(session_id, tenant_id=tenant_id, user_id=user_id)
        if history is None:
            return []
        with history._lock:
            return [dict(item) for item in history.traces[-max(0, limit):]]

    def clear_session(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        scope = self._scope(session_id, tenant_id, user_id)
        if self.storage_type == "memory":
            with self._lock:
                self.sessions.pop(scope.cache_key, None)
            return
        if self.redis_client is not None:
            self.redis_client.delete(*self._redis_keys(scope).values())

    async def aadd_message(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self.add_message, *args, **kwargs)

    async def aget_history(self, *args: Any, **kwargs: Any) -> List[Dict[str, str]]:
        return await asyncio.to_thread(self.get_history, *args, **kwargs)

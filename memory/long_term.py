"""长期记忆边界。

Mem0 只是可选的外部存储适配器。本模块负责用户隔离、授权、
结构化记录、过期和检索后过滤；不将旧 AI 回答冒充医学事实。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import hashlib
import json
import os
import uuid

from loguru import logger

from .entropy_manager import MemoryEntropyManager


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _strict_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return default


@dataclass
class LongTermMemoryRecord:
    """可审计的长期记忆记录。"""

    tenant_id: str
    user_id: str
    session_id: str
    question_summary: str
    answer_summary: str
    turn_id: Optional[str] = None
    facts: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "assistant_generated"
    confidence: float = 0.5
    consent: bool = False
    contains_phi: bool = True
    created_at: datetime = field(default_factory=_utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 2

    def __post_init__(self) -> None:
        for name in ("tenant_id", "user_id", "session_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.created_at = _parse_datetime(self.created_at) or _utcnow()
        self.expires_at = _parse_datetime(self.expires_at)
        self.facts = [dict(item) for item in self.facts if isinstance(item, Mapping)]
        self.metadata = dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LongTermMemoryRecord":
        allowed = {
            "tenant_id", "user_id", "session_id", "question_summary", "answer_summary",
            "turn_id", "facts", "source", "confidence", "consent", "contains_phi",
            "created_at", "expires_at", "metadata", "record_id", "schema_version",
        }
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)  # type: ignore[arg-type]


class LongTermMemory:
    """外部长期记忆适配器。

    ``allow_external_phi`` 必须在系统配置中显式开启，每次写入还必须
    传入 ``consent=True``。默认调用会安全地拒绝上传可能含 PHI 的内容。
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        client: Any = None,
        default_tenant_id: str = "default",
        default_user_id: str = "anonymous",
        allow_external_phi: bool = False,
        min_similarity_score: float = 0.55,
        default_retention_days: int = 90,
        max_memory_chars: int = 8000,
        entropy_manager: Optional[MemoryEntropyManager] = None,
    ) -> None:
        settings = dict(config or {})
        self.default_tenant_id = str(settings.get("tenant_id", default_tenant_id))
        self.default_user_id = str(settings.get("user_id", default_user_id))
        self.allow_external_phi = _strict_bool(
            settings.get("allow_external_phi", allow_external_phi),
            default=False,
        )
        self.min_similarity_score = float(settings.get("min_similarity_score", min_similarity_score))
        self.default_retention_days = int(settings.get("retention_days", default_retention_days))
        self.max_memory_chars = int(settings.get("max_memory_chars", max_memory_chars))
        self.entropy_manager = entropy_manager or MemoryEntropyManager()
        if not self.default_tenant_id or not self.default_user_id:
            raise ValueError("default tenant and user IDs must be non-empty")
        if not 0.0 <= self.min_similarity_score <= 1.0:
            raise ValueError("min_similarity_score must be between 0 and 1")
        if self.default_retention_days <= 0 or self.max_memory_chars < 2048:
            raise ValueError("retention and memory size limits must be positive")

        self.mem0 = client
        if self.mem0 is None:
            api_key = settings.get("api_key") or os.getenv("MEM0_API_KEY")
            if api_key:
                try:
                    from mem0 import MemoryClient  # type: ignore

                    self.mem0 = MemoryClient(api_key=api_key)
                except Exception as exc:
                    logger.warning(
                        "Long-term memory client unavailable: {}",
                        type(exc).__name__,
                    )
        self.enabled = self.mem0 is not None

    @staticmethod
    def _validate_scope(tenant_id: str, user_id: str) -> None:
        for name, value in (("tenant_id", tenant_id), ("user_id", user_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if len(value) > 512:
                raise ValueError(f"{name} is too long")

    @staticmethod
    def _backend_user_id(tenant_id: str, user_id: str) -> str:
        # 后端不需看到原始账号标识；哈希同时消除分隔符冲突。
        payload = json.dumps([tenant_id, user_id], ensure_ascii=False, separators=(",", ":"))
        return "medix_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _scope_token(kind: str, *values: str) -> str:
        payload = json.dumps([kind, *values], ensure_ascii=False, separators=(",", ":"))
        return f"{kind}_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _safe_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """确保元数据可 JSON 化，避免客户端在远端报错。"""
        try:
            # 作用域标识由本模块生成不可逆 token；不接受调用方把原始账号/
            # 会话标识再次塞进外部元数据。
            reserved = {"tenant_id", "user_id", "session_id", "turn_id", "patient_id"}
            cleaned = {
                str(key): value
                for key, value in dict(metadata or {}).items()
                if str(key).lower() not in reserved
            }
            serialisable = json.loads(json.dumps(cleaned, ensure_ascii=False, default=str))
            encoded = json.dumps(serialisable, ensure_ascii=False, separators=(",", ":"))
            if len(encoded) <= 4000:
                return serialisable
            # 保留体积可控的标量字段，防止元数据绕过记忆载荷预算。
            compact = {
                str(key): value
                for key, value in serialisable.items()
                if isinstance(value, (str, int, float, bool, type(None)))
                and len(str(value)) <= 512
            }
            compact["metadata_compacted"] = True
            return compact
        except (TypeError, ValueError):
            return {}

    def _bounded(self, value: str, budget: int) -> str:
        text = str(value or "").strip()
        if len(text) <= budget:
            return text
        marker = "……[长期记忆预算裁剪]"
        return text[:max(0, budget - len(marker))] + marker

    def _serialise_record(self, record: LongTermMemoryRecord) -> str:
        """始终返回有效 JSON，不在任意字节位置截断结构化载荷。"""
        payload = record.to_dict()

        def dumps() -> str:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

        encoded = dumps()
        if len(encoded) <= self.max_memory_chars:
            return encoded

        # 大型扩展元数据和事实不得挤破基本记忆结构；完整版仍可保留在本地 SessionSummary。
        payload["facts"] = []
        payload["metadata"] = {"payload_compacted": True}
        payload["question_summary"] = ""
        payload["answer_summary"] = ""
        overhead = len(dumps())
        available = max(256, self.max_memory_chars - overhead - 32)
        question_budget = max(128, available // 3)
        answer_budget = max(128, available - question_budget)
        payload["question_summary"] = self._bounded(record.question_summary, question_budget)
        payload["answer_summary"] = self._bounded(record.answer_summary, answer_budget)
        encoded = dumps()
        # JSON 转义可能令字符数略超估算，逐步收紧两个可变字段。
        while len(encoded) > self.max_memory_chars and (
            payload["question_summary"] or payload["answer_summary"]
        ):
            overflow = len(encoded) - self.max_memory_chars + 16
            field_name = (
                "answer_summary"
                if len(payload["answer_summary"]) >= len(payload["question_summary"])
                else "question_summary"
            )
            current = str(payload[field_name])
            payload[field_name] = current[:max(0, len(current) - overflow)]
            encoded = dumps()
        return encoded

    def add_session_summary(
        self,
        session_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        facts: Optional[Sequence[Mapping[str, Any]]] = None,
        source: str = "assistant_generated",
        confidence: float = 0.5,
        consent: bool = False,
        contains_phi: bool = True,
        expires_at: Optional[datetime] = None,
    ) -> Optional[str]:
        """保存结构化会话记忆。

        ``facts`` 中每项建议包含 ``value/source/confidence/observed_at``。
        这些事实仍是用户陈述或模型提取，不会被标记为医学知识。
        """
        if not self.enabled:
            return None
        tenant = tenant_id or self.default_tenant_id
        user = user_id or self.default_user_id
        self._validate_scope(tenant, user)
        if not isinstance(consent, bool):
            raise TypeError("consent must be a boolean")
        if contains_phi and not (self.allow_external_phi and consent):
            logger.warning(
                "Skipped external long-term memory write: PHI requires both "
                "allow_external_phi=True and per-write consent=True"
            )
            return None

        expiry = _parse_datetime(expires_at) or (_utcnow() + timedelta(days=self.default_retention_days))
        per_field_budget = max(128, self.max_memory_chars // 2)
        safe_facts = json.loads(json.dumps(
            [dict(item) for item in (facts or ()) if isinstance(item, Mapping)],
            ensure_ascii=False,
            default=str,
        ))
        record = LongTermMemoryRecord(
            tenant_id=tenant,
            user_id=user,
            session_id=session_id,
            turn_id=turn_id,
            question_summary=self._bounded(question, per_field_budget),
            answer_summary=self._bounded(answer, per_field_budget),
            facts=safe_facts,
            source=source,
            confidence=float(confidence),
            consent=consent,
            contains_phi=contains_phi,
            expires_at=expiry,
            metadata=self._safe_metadata(metadata),
        )
        scope_id = self._backend_user_id(tenant, user)
        session_scope = self._scope_token("session", tenant, user, session_id)
        turn_scope = self._scope_token("turn", tenant, user, session_id, turn_id) if turn_id else None
        # 外部载荷不携带原始租户、用户、会话或回合标识。
        external_record = replace(
            record,
            tenant_id=self._scope_token("tenant", tenant),
            user_id=self._scope_token("user", tenant, user),
            session_id=session_scope,
            turn_id=turn_scope,
        )
        memory_text = self._serialise_record(external_record)
        authoritative_metadata = {
            **record.metadata,
            "schema_version": record.schema_version,
            "type": "session_summary",
            "record_id": record.record_id,
            "scope_id": scope_id,
            "session_scope": session_scope,
            "turn_scope": turn_scope,
            "source": source,
            "confidence": record.confidence,
            "consent": consent,
            "contains_phi": contains_phi,
            "timestamp": record.created_at.isoformat(),
            "expires_at": expiry.isoformat(),
        }
        try:
            result = self.mem0.add(
                messages=[{"role": "user", "content": memory_text}],
                user_id=scope_id,
                metadata=authoritative_metadata,
            )
            if isinstance(result, Mapping):
                direct = result.get("id")
                if direct:
                    return str(direct)
                results = result.get("results") or []
                if results and isinstance(results[0], Mapping) and results[0].get("id"):
                    return str(results[0]["id"])
            if result is not None:
                return str(result)
        except Exception as exc:
            logger.error("Failed to add long-term memory: {}", type(exc).__name__)
        return None

    @staticmethod
    def _result_list(results: Any) -> List[Mapping[str, Any]]:
        if isinstance(results, Mapping):
            candidates = results.get("results") or results.get("memories") or []
            return [item for item in candidates if isinstance(item, Mapping)]
        if isinstance(results, list):
            return [item for item in results if isinstance(item, Mapping)]
        return []

    @staticmethod
    def _parse_record(content: str) -> Optional[LongTermMemoryRecord]:
        try:
            payload = json.loads(content)
            if isinstance(payload, Mapping) and int(payload.get("schema_version", 0)) >= 2:
                return LongTermMemoryRecord.from_dict(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return None

    def search_similar_sessions(
        self,
        query: str,
        limit: int = 5,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_score: Optional[float] = None,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        current_session_id: Optional[str] = None,
        exclude_memory_ids: Optional[Iterable[str]] = None,
        include_expired: bool = False,
        consent: bool = False,
        contains_phi: bool = True,
    ) -> List[Dict[str, Any]]:
        """检索后再执行服务端用户隔离和过期校验。"""
        if not self.enabled or limit <= 0 or not str(query).strip():
            return []
        tenant = tenant_id or self.default_tenant_id
        user = user_id or self.default_user_id
        self._validate_scope(tenant, user)
        if not isinstance(consent, bool):
            raise TypeError("consent must be a boolean")
        if contains_phi and not (self.allow_external_phi and consent):
            logger.warning(
                "Skipped external long-term memory search: PHI requires both "
                "allow_external_phi=True and per-search consent=True"
            )
            return []
        score_floor = self.min_similarity_score if min_score is None else float(min_score)
        if not 0.0 <= score_floor <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        requested = min(max(limit * 4, limit), 100)
        scope_id = self._backend_user_id(tenant, user)
        current_session_scope = (
            self._scope_token("session", tenant, user, current_session_id)
            if current_session_id else None
        )
        safe_filters = {
            str(key): value
            for key, value in dict(metadata_filter or {}).items()
            if str(key).lower() not in {
                "tenant_id", "user_id", "session_id", "turn_id", "patient_id", "scope_id"
            }
        }
        backend_filters = {**safe_filters, "type": "session_summary"}
        kwargs = {
            "query": str(query),
            "user_id": self._backend_user_id(tenant, user),
            "limit": requested,
        }
        try:
            try:
                results = self.mem0.search(**kwargs, filters=backend_filters)
            except TypeError:
                # 兼容不支持 filters 参数的旧客户端；下方仍会强制本地过滤。
                results = self.mem0.search(**kwargs)
        except Exception as exc:
            logger.error("Failed to search long-term memory: {}", type(exc).__name__)
            return []

        formatted: List[Dict[str, Any]] = []
        for result in self._result_list(results):
            metadata = dict(result.get("metadata") or {})
            # Mem0 的 user_id 已形成第一层隔离；这里再校验不可逆 scope token。
            # 兼容同一哈希用户空间中的旧版原始 metadata，便于安全迁移。
            owns_record = metadata.get("scope_id") == scope_id or (
                metadata.get("tenant_id") == tenant and metadata.get("user_id") == user
            )
            if not owns_record:
                continue
            session_scope = metadata.get("session_scope") or metadata.get("session_id")
            if current_session_scope and session_scope in {current_session_scope, current_session_id}:
                continue
            content = str(result.get("memory") or result.get("text") or result.get("content") or "")
            try:
                score = float(result.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            record = self._parse_record(content)
            formatted.append({
                "memory_id": str(result.get("id") or metadata.get("record_id") or "unknown"),
                "content": content,
                "score": score,
                "metadata": metadata,
                "timestamp": metadata.get("timestamp"),
                "expires_at": metadata.get("expires_at"),
                "session_id": session_scope,
                "record": record.to_dict() if record else None,
            })

        filtered = self.entropy_manager.deduplicate_sessions(
            formatted,
            min_score=score_floor,
            metadata_filter=safe_filters,
            tenant_id=None,
            user_id=None,
            current_session_id=None,
            exclude_memory_ids=exclude_memory_ids,
            include_expired=include_expired,
        )
        # 不信任外部存储的默认排序。
        filtered.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return filtered[:limit]

    def delete_memory(
        self,
        memory_id: str,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """删除指定记忆，供用户撤回授权/数据删除流程调用。"""
        if not self.enabled or not memory_id:
            return False
        tenant = tenant_id or self.default_tenant_id
        user = user_id or self.default_user_id
        self._validate_scope(tenant, user)
        # 先在当前用户空间查询，防止用户提供他人 ID 越权删除。
        try:
            getter = getattr(self.mem0, "get", None)
            if not getter:
                # 无法先验证所属权时，宁可拒绝也不越权删除。
                return False
            record = getter(memory_id=memory_id)
            metadata = dict(record.get("metadata") or {}) if isinstance(record, Mapping) else {}
            scope_id = self._backend_user_id(tenant, user)
            owns_record = metadata.get("scope_id") == scope_id or (
                metadata.get("tenant_id") == tenant and metadata.get("user_id") == user
            )
            if not owns_record:
                return False
            deleter = getattr(self.mem0, "delete", None)
            if not deleter:
                return False
            try:
                deleter(memory_id=memory_id)
            except TypeError:
                deleter(memory_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete long-term memory: {}", type(exc).__name__)
            return False

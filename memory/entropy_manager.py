"""记忆清理与预算管理。

这里只做确定性、可审计的清理，不把字符串截断伪装成模型摘要。
语义摘要应由上层在明确的模型和隐私策略下生成。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import re

from loguru import logger


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class MemoryEntropyManager:
    """无外部依赖的记忆卫生管理器。

    ``deduplication_threshold`` 用于长期记忆的近似去重；短期对话
    默认只按稳定的 ``message_id`` 或完全相同内容去重，以免删掉
    用户真正重复询问的不同回合。
    """

    def __init__(
        self,
        deduplication_threshold: float = 0.92,
        max_age_days: int = 90,
        compression_threshold: int = 20,
    ) -> None:
        if not 0.0 <= deduplication_threshold <= 1.0:
            raise ValueError("deduplication_threshold must be between 0 and 1")
        if max_age_days <= 0 or compression_threshold <= 0:
            raise ValueError("age and compression thresholds must be positive")
        self.deduplication_threshold = deduplication_threshold
        self.max_age_days = max_age_days
        self.compression_threshold = compression_threshold

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """保守的 token 估算，无需 tokenizer。

        中文字符大致按 1 token，连续的拉丁文本大致按 4 字符/token。
        这是预算上界工具，不是计费依据。
        """
        if not text:
            return 0
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        non_cjk = max(0, len(text) - cjk)
        return cjk + (non_cjk + 3) // 4

    @staticmethod
    def _normalise_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        return re.sub(r"\s+", " ", text)

    @classmethod
    def _session_content(cls, session: Mapping[str, Any]) -> str:
        """兼容 Mem0 格式和项目旧格式，避免字段不匹配导致全部哈希为空。"""
        direct = session.get("content") or session.get("memory") or session.get("text")
        if direct:
            return cls._normalise_text(direct)

        pieces = [
            session.get("question"),
            session.get("question_summary"),
            session.get("summary"),
            session.get("answer"),
            session.get("answer_summary"),
            session.get("final_answer"),
        ]
        return cls._normalise_text("\n".join(str(item) for item in pieces if item))

    @staticmethod
    def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = metadata.get(key)
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def deduplicate_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """对完全重复的消息去重，并保留时间上最新的一条。

        若消息有 ``message_id``，以它为准；否则以 role/content/type 组合
        为准。返回结果仍保持原始时序。
        """
        seen: set[str] = set()
        kept_reversed: List[Dict[str, Any]] = []
        for message in reversed(messages):
            msg = dict(message)
            message_id = msg.get("message_id")
            if message_id:
                identity = f"id:{message_id}"
            else:
                payload = {
                    "role": msg.get("role", ""),
                    "content": self._normalise_text(msg.get("content", "")),
                    "message_type": msg.get("message_type", "dialogue"),
                }
                identity = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
            if identity not in seen:
                kept_reversed.append(msg)
                seen.add(identity)
        return list(reversed(kept_reversed))

    def deduplicate_sessions(
        self,
        sessions: Sequence[Mapping[str, Any]],
        similarity_threshold: Optional[float] = None,
        min_score: Optional[float] = None,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        current_session_id: Optional[str] = None,
        exclude_memory_ids: Optional[Iterable[str]] = None,
        include_expired: bool = False,
        include_empty: bool = False,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """过滤并去重长期记忆检索结果。

        顺序是：用户/元数据/当前会话/过期/分数过滤，再按内容相似度
        去重。输入顺序会被保留，因此调用者应先按相似度降序排列。
        """
        threshold = self.deduplication_threshold if similarity_threshold is None else similarity_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        excluded_ids = {str(item) for item in (exclude_memory_ids or ())}
        filters = dict(metadata_filter or {})
        if tenant_id is not None:
            filters["tenant_id"] = tenant_id
        if user_id is not None:
            filters["user_id"] = user_id
        current_time = now or _utcnow()

        unique: List[Dict[str, Any]] = []
        unique_contents: List[str] = []
        exact_hashes: set[str] = set()

        for raw in sessions:
            session = dict(raw)
            metadata = session.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            effective_metadata = {
                key: session.get(key)
                for key in ("tenant_id", "user_id", "session_id", "type", "source")
                if session.get(key) is not None
            }
            effective_metadata.update(metadata)
            memory_id = str(session.get("memory_id") or session.get("id") or "")
            if memory_id and memory_id in excluded_ids:
                continue
            session_id = session.get("session_id") or effective_metadata.get("session_id")
            if current_session_id is not None and session_id == current_session_id:
                continue
            if filters and not self._metadata_matches(effective_metadata, filters):
                continue

            score = session.get("score")
            try:
                numeric_score = float(score) if score is not None else None
            except (TypeError, ValueError):
                numeric_score = None
            if min_score is not None and (numeric_score is None or numeric_score < min_score):
                continue

            expires_at = session.get("expires_at") or effective_metadata.get("expires_at")
            expiry = _as_aware(expires_at)
            if not include_expired and expiry is not None and expiry <= current_time:
                continue

            content = self._session_content(session)
            if not content:
                if include_empty:
                    unique.append(session)
                    unique_contents.append("")
                continue

            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest in exact_hashes:
                continue
            if any(
                previous
                and SequenceMatcher(None, content, previous, autojunk=False).ratio() >= threshold
                for previous in unique_contents
            ):
                continue

            exact_hashes.add(digest)
            unique.append(session)
            unique_contents.append(content)

        return unique

    def cleanup_old_memories(
        self,
        memories: Sequence[Mapping[str, Any]],
        max_age_days: Optional[int] = None,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """清理过期记忆；显式 ``expires_at`` 优先于默认保留期。"""
        age_days = self.max_age_days if max_age_days is None else max_age_days
        if age_days <= 0:
            raise ValueError("max_age_days must be positive")
        current_time = now or _utcnow()
        cutoff = current_time - timedelta(days=age_days)
        cleaned: List[Dict[str, Any]] = []
        for raw in memories:
            memory = dict(raw)
            metadata = memory.get("metadata") or {}
            explicit_expiry = _as_aware(memory.get("expires_at") or metadata.get("expires_at"))
            if explicit_expiry is not None:
                if explicit_expiry > current_time:
                    cleaned.append(memory)
                continue
            timestamp = _as_aware(memory.get("timestamp") or metadata.get("timestamp"))
            if timestamp is None or timestamp > cutoff:
                cleaned.append(memory)
        return cleaned

    @staticmethod
    def complete_turns(
        messages: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """将消息拆成完整 user/assistant 回合，忽略 tool/trace/system。

        同一 user 后有多条 assistant 时取下一个 user 之前最后的非轨迹
        assistant，因而不会把“调用工具”之类中间态当成最终回答。
        """
        turns: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        current_user: Optional[Dict[str, Any]] = None
        current_assistant: Optional[Dict[str, Any]] = None
        for raw in messages:
            message = dict(raw)
            role = message.get("role")
            message_type = message.get("message_type", "dialogue")
            if message_type in {"tool", "trace", "internal"} or role in {"tool", "function"}:
                continue
            if role == "user":
                if current_user is not None and current_assistant is not None:
                    turns.append((current_user, current_assistant))
                current_user = message
                current_assistant = None
            elif role == "assistant" and current_user is not None:
                if not str(message.get("content", "")).startswith("调用工具："):
                    current_assistant = message
        if current_user is not None and current_assistant is not None:
            turns.append((current_user, current_assistant))
        return turns

    @staticmethod
    def _deterministic_summary(
        turns: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
        char_budget: int,
    ) -> str:
        """生成明确标记为“摘录”的滚动文本，不声称语义无损。"""
        if char_budget <= 0:
            return ""
        lines: List[str] = []
        for user, assistant in turns:
            question = str(user.get("content", "")).strip().replace("\n", " ")[:240]
            answer = str(assistant.get("content", "")).strip().replace("\n", " ")[:360]
            lines.append(f"- 用户：{question}\n  助手：{answer}")
        text = "\n".join(lines)
        if len(text) <= char_budget:
            return text
        # 优先保留较新的旧回合。
        return "……（更早摘录已按预算裁剪）\n" + text[-max(0, char_budget - 20):]

    def compress_session_history(
        self,
        messages: Sequence[Mapping[str, Any]],
        max_messages: int = 10,
        char_budget: Optional[int] = None,
        existing_summary: str = "",
    ) -> List[Dict[str, Any]]:
        """按完整回合保留最近历史，旧回合转为明确的系统摘录。"""
        if max_messages < 2:
            raise ValueError("max_messages must allow at least one complete turn")
        turns = self.complete_turns(messages)
        keep_turns = max(1, max_messages // 2)
        recent = turns[-keep_turns:]
        older = turns[:-keep_turns]
        summary_budget = max(256, (char_budget or 4000) // 3)
        extracted = self._deterministic_summary(older, summary_budget)
        summaries = [part for part in (existing_summary.strip(), extracted.strip()) if part]
        result: List[Dict[str, Any]] = []
        if summaries:
            summary = "\n".join(summaries)
            result.append({
                "role": "system",
                "content": "[历史回合摘录，非原文]\n" + summary[-summary_budget:],
                "message_type": "summary",
            })
        for user, assistant in recent:
            result.extend((dict(user), dict(assistant)))

        if char_budget is not None:
            while len(result) > 2 and sum(len(str(m.get("content", ""))) for m in result) > char_budget:
                # 摘要之后每次删一个完整回合。
                offset = 1 if result and result[0].get("message_type") == "summary" else 0
                del result[offset:offset + 2]
        return result

    def estimate_entropy(self, messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not messages:
            return {
                "total_messages": 0,
                "unique_messages": 0,
                "estimated_duplicates": 0,
                "duplicate_rate": 0.0,
                "avg_message_length": 0.0,
                "estimated_tokens": 0,
                "complete_turns": 0,
                "entropy_level": "low",
                "recommendations": [],
            }
        normalised = [
            f"{m.get('role', '')}:{self._normalise_text(m.get('content', ''))}"
            for m in messages
        ]
        unique_count = len(set(normalised))
        total = len(messages)
        duplicate_count = total - unique_count
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        estimated_tokens = sum(self.estimate_tokens(str(m.get("content", ""))) for m in messages)
        complete_turn_count = len(self.complete_turns(messages))
        duplicate_rate = duplicate_count / total
        level = "high" if total > 50 or total_chars > 30000 else "medium" if total > 20 or total_chars > 12000 else "low"
        recommendations: List[str] = []
        if level != "low":
            recommendations.append("按完整回合压缩历史并保留滚动摘要")
        if duplicate_rate > 0.2:
            recommendations.append("检查是否重复写入同一消息")
        return {
            "total_messages": total,
            "unique_messages": unique_count,
            "estimated_duplicates": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "avg_message_length": total_chars / total,
            "estimated_tokens": estimated_tokens,
            "complete_turns": complete_turn_count,
            "entropy_level": level,
            "recommendations": recommendations,
        }

    def auto_clean(
        self,
        messages: Sequence[Mapping[str, Any]],
        enable_deduplication: bool = True,
        enable_compression: bool = True,
        max_messages: int = 10,
        char_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        cleaned = self.deduplicate_messages(messages) if enable_deduplication else [dict(m) for m in messages]
        if enable_compression and (
            len(cleaned) > max_messages
            or (char_budget is not None and sum(len(str(m.get("content", ""))) for m in cleaned) > char_budget)
        ):
            cleaned = self.compress_session_history(cleaned, max_messages, char_budget)
        return cleaned

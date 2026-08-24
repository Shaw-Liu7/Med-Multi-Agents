"""Swarm 会话总结的可逆、本地持久化。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping, Optional, Sequence
import base64
import hashlib
import json
import os
import re
import tempfile
import uuid

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return _utcnow()


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class AgentParticipation:
    agent_id: str
    role: str
    subtasks_handled: List[str]
    tool_calls: int
    execution_time: float
    contribution_quality: float = 1.0


@dataclass
class KeyFinding:
    category: str
    finding: str
    source_agent: str
    confidence: float = 1.0


@dataclass
class Lesson:
    agent_id: str
    lesson_type: str
    description: str
    actionable: str


@dataclass
class PerformanceMetrics:
    total_time: float
    agent_count: int
    parallel_efficiency: float
    information_coverage: float
    redundancy: float
    speedup_vs_single: float = 1.0
    completed_subtask_ratio: float = 0.0
    total_worker_time: float = 0.0


@dataclass
class SessionSummary:
    """一次 Swarm 执行的完整记录。"""

    session_id: str
    question: str
    context: Dict[str, Any]
    timestamp: datetime
    agents_participated: List[AgentParticipation]
    subtasks_created: int
    subtasks_completed: int
    events_count: int
    final_answer: str
    key_findings: List[KeyFinding]
    lessons_learned: List[Lesson]
    performance: PerformanceMetrics
    swarm_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"
    user_id: str = "anonymous"
    turn_id: Optional[str] = None
    summary_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 2

    def __post_init__(self) -> None:
        self.timestamp = _parse_datetime(self.timestamp)
        for name in ("tenant_id", "user_id", "session_id", "summary_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionSummary":
        agents = [AgentParticipation(**dict(item)) for item in data.get("agents_participated", [])]
        findings = [KeyFinding(**dict(item)) for item in data.get("key_findings", [])]
        lessons = [Lesson(**dict(item)) for item in data.get("lessons_learned", [])]
        performance_data = dict(data.get("performance") or {})
        performance = PerformanceMetrics(**performance_data)
        return cls(
            session_id=str(data["session_id"]),
            question=str(data.get("question", "")),
            context=dict(data.get("context") or {}),
            timestamp=_parse_datetime(data.get("timestamp")),
            agents_participated=agents,
            subtasks_created=int(data.get("subtasks_created", 0)),
            subtasks_completed=int(data.get("subtasks_completed", 0)),
            events_count=int(data.get("events_count", 0)),
            final_answer=str(data.get("final_answer", "")),
            key_findings=findings,
            lessons_learned=lessons,
            performance=performance,
            swarm_enabled=bool(data.get("swarm_enabled", True)),
            metadata=dict(data.get("metadata") or {}),
            tenant_id=str(data.get("tenant_id") or "default"),
            user_id=str(data.get("user_id") or "anonymous"),
            turn_id=str(data["turn_id"]) if data.get("turn_id") is not None else None,
            summary_id=str(data.get("summary_id") or uuid.uuid4()),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_markdown(self) -> str:
        """生成可读 Markdown，并嵌入 Base64 JSON 作为无损回读载荷。"""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        lines = [
            f"# Session Summary: {self.session_id}",
            "",
            f"**时间**: {self.timestamp.isoformat()}",
            f"**总结 ID**: {self.summary_id}",
            f"**回合 ID**: {self.turn_id or '-'}",
            "",
            "## 问题",
            "",
            self.question,
            "",
        ]
        if self.context:
            lines.extend([
                "## 背景",
                "",
                "```json",
                json.dumps(self.context, ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ])
        lines.extend(["## 参与 Agent", ""])
        for agent in self.agents_participated:
            lines.extend([
                f"### {agent.agent_id} ({agent.role})",
                f"- 处理子任务：{len(agent.subtasks_handled)} 个",
                f"- 工具调用：{agent.tool_calls} 次",
                f"- 可观测执行时间：{agent.execution_time:.3f} 秒",
                f"- 贡献置信度：{agent.contribution_quality:.1%}",
                "",
            ])
        lines.extend([
            "## 协作过程",
            "",
            f"- 创建子任务：{self.subtasks_created} 个",
            f"- 完成子任务：{self.subtasks_completed} 个",
            f"- 发布事件：{self.events_count} 个",
            "",
        ])
        if self.key_findings:
            lines.extend(["## 关键发现", ""])
            for finding in self.key_findings:
                lines.extend([
                    f"### {finding.category.upper()}",
                    f"**来源**: {finding.source_agent}",
                    f"**发现**: {finding.finding}",
                    f"**置信度**: {finding.confidence:.1%}",
                    "",
                ])
        lines.extend(["## 最终答案", "", self.final_answer, ""])
        if self.lessons_learned:
            lines.extend(["## 经验教训", ""])
            for lesson in self.lessons_learned:
                lines.extend([
                    f"### {lesson.agent_id} / {lesson.lesson_type}",
                    lesson.description,
                    f"**可执行改进**: {lesson.actionable}" if lesson.actionable else "",
                    "",
                ])
        metric = self.performance
        lines.extend([
            "## 性能指标",
            "",
            f"- 总耗时：{metric.total_time:.3f} 秒",
            f"- 参与 Agent：{metric.agent_count} 个",
            f"- 子任务完成率：{metric.completed_subtask_ratio:.1%}",
            f"- 并行利用率：{metric.parallel_efficiency:.1%}",
            f"- 信息覆盖度：{metric.information_coverage:.1%}",
            f"- 结果冗余度：{metric.redundancy:.1%}",
            f"- 可观测工作量/墙钟时间：{metric.speedup_vs_single:.2f}x",
            "",
            "<!-- MEDIX_SESSION_SUMMARY_V2",
            encoded,
            "-->",
            "",
        ])
        return "\n".join(line for line in lines if line is not None)

    @classmethod
    def from_markdown(cls, markdown_content: str) -> "SessionSummary":
        match = re.search(
            r"<!--\s*MEDIX_SESSION_SUMMARY_V2\s*\n([A-Za-z0-9+/=\s]+?)\n-->",
            markdown_content,
            flags=re.MULTILINE,
        )
        if not match:
            raise ValueError("Markdown does not contain a reversible SessionSummary payload")
        encoded = re.sub(r"\s+", "", match.group(1))
        try:
            payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid SessionSummary payload") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("SessionSummary payload must be a JSON object")
        return cls.from_dict(payload)

    @staticmethod
    def _result_similarity(values: Sequence[str]) -> float:
        """用 token Jaccard 的平均值估算结果凗余；样本少于 2 时为 0。"""
        token_sets = [set(re.findall(r"[\w\u3400-\u9fff]+", value.lower())) for value in values if value]
        if len(token_sets) < 2:
            return 0.0
        similarities: List[float] = []
        for index, left in enumerate(token_sets):
            for right in token_sets[index + 1:]:
                union = left | right
                similarities.append(len(left & right) / len(union) if union else 0.0)
        return sum(similarities) / len(similarities) if similarities else 0.0

    @classmethod
    def from_shared_context(
        cls,
        session_id: str,
        question: str,
        shared_context: Any,
        final_answer: str,
        start_time: datetime,
        end_time: datetime,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        turn_id: Optional[str] = None,
    ) -> "SessionSummary":
        """仅用 SharedContext 中真实可观测字段计算指标。"""
        start = _parse_datetime(start_time)
        end = _parse_datetime(end_time)
        total_time = max(0.0, (end - start).total_seconds())
        tasks = list(getattr(shared_context, "task_decomposition", {}).values())
        contributions_by_agent = dict(getattr(shared_context, "agent_contributions", {}))
        contributions = list(shared_context.get_contributions()) if hasattr(shared_context, "get_contributions") else [
            contribution for values in contributions_by_agent.values() for contribution in values
        ]

        durations_by_agent: Dict[str, float] = {}
        handled_by_agent: Dict[str, List[str]] = {}
        completed_count = 0
        failed_tasks: List[Any] = []
        for task in tasks:
            status_value = getattr(getattr(task, "status", None), "value", str(getattr(task, "status", "")))
            if status_value == "completed":
                completed_count += 1
            elif status_value == "failed":
                failed_tasks.append(task)
            agent_id = str(getattr(task, "assigned_agent", None) or getattr(task, "assigned_to", None) or "unknown")
            handled_by_agent.setdefault(agent_id, []).append(str(getattr(task, "id", "unknown")))
            started = getattr(task, "started_at", None)
            completed = getattr(task, "completed_at", None)
            if started is not None and completed is not None:
                duration = max(0.0, (_parse_datetime(completed) - _parse_datetime(started)).total_seconds())
                durations_by_agent[agent_id] = durations_by_agent.get(agent_id, 0.0) + duration

        all_agent_ids = set(handled_by_agent) | set(contributions_by_agent)
        all_agent_ids.discard("unknown")
        agents: List[AgentParticipation] = []
        for agent_id in sorted(all_agent_ids):
            agent_contributions = list(contributions_by_agent.get(agent_id, []))
            tool_calls = 0
            confidences: List[float] = []
            for contribution in agent_contributions:
                result = getattr(contribution, "result", {}) or {}
                metadata = getattr(contribution, "metadata", {}) or {}
                explicit_calls = metadata.get("tool_calls", result.get("tool_calls", 0))
                if isinstance(explicit_calls, list):
                    tool_calls += len(explicit_calls)
                else:
                    try:
                        tool_calls += max(0, int(explicit_calls))
                    except (TypeError, ValueError):
                        pass
                confidences.append(_bounded_float(getattr(contribution, "confidence", 0.0)))
            agents.append(AgentParticipation(
                agent_id=agent_id,
                role="lead" if agent_id == "lead_agent" else "worker",
                subtasks_handled=handled_by_agent.get(agent_id, []),
                tool_calls=tool_calls,
                execution_time=durations_by_agent.get(agent_id, 0.0),
                contribution_quality=sum(confidences) / len(confidences) if confidences else 0.0,
            ))

        findings: List[KeyFinding] = []
        result_texts: List[str] = []
        for contribution in contributions:
            result = getattr(contribution, "result", {}) or {}
            agent_id = str(getattr(contribution, "agent_id", "unknown"))
            confidence = _bounded_float(getattr(contribution, "confidence", 0.0))
            result_texts.append(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
            for key, category in (("risk_level", "risk"), ("diagnosis", "diagnosis"), ("evidence", "evidence")):
                if result.get(key) not in (None, "", [], {}):
                    findings.append(KeyFinding(category, str(result[key]), agent_id, confidence))

        lessons: List[Lesson] = []
        for task in failed_tasks:
            result = getattr(task, "result", {}) or {}
            agent_id = str(getattr(task, "assigned_agent", "unknown"))
            lessons.append(Lesson(
                agent_id=agent_id,
                lesson_type="failure",
                description=str(result.get("error") or f"子任务 {getattr(task, 'id', 'unknown')} 失败"),
                actionable="复查输入、工具依赖和超时策略后重试。",
            ))

        task_count = len(tasks)
        completion_ratio = completed_count / task_count if task_count else 0.0
        worker_time = sum(durations_by_agent.values())
        agent_count = len(all_agent_ids)
        parallel_efficiency = (
            min(1.0, worker_time / (total_time * agent_count))
            if total_time > 0 and agent_count > 0 else 0.0
        )
        speedup = worker_time / total_time if total_time > 0 else 0.0
        performance = PerformanceMetrics(
            total_time=total_time,
            agent_count=agent_count,
            parallel_efficiency=parallel_efficiency,
            information_coverage=completion_ratio,
            redundancy=cls._result_similarity(result_texts),
            speedup_vs_single=speedup,
            completed_subtask_ratio=completion_ratio,
            total_worker_time=worker_time,
        )
        return cls(
            session_id=session_id,
            question=question,
            context=dict(getattr(shared_context, "data", {}) or {}),
            timestamp=start,
            agents_participated=agents,
            subtasks_created=task_count,
            subtasks_completed=completed_count,
            events_count=len(getattr(shared_context, "events", [])),
            final_answer=final_answer,
            key_findings=findings,
            lessons_learned=lessons,
            performance=performance,
            swarm_enabled=task_count > 0,
            tenant_id=tenant_id,
            user_id=user_id,
            turn_id=turn_id,
            metadata={"metrics_source": "shared_context_observations"},
        )


class SessionSummaryManager:
    """以 JSON 为权威格式、Markdown 为人类可读副本的管理器。"""

    def __init__(self, base_dir: str = "memory/swarm/session_summaries") -> None:
        requested = Path(base_dir).expanduser()
        self.base_dir = (requested if requested.is_absolute() else PROJECT_ROOT / requested).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:
            pass
        self._lock = RLock()

    @staticmethod
    def _safe_component(value: str, prefix: str = "item") -> str:
        raw = str(value)
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")[:48] or prefix
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{cleaned}-{digest}"

    @staticmethod
    def _scope_digest(tenant_id: str, user_id: str, session_id: str) -> str:
        payload = json.dumps([tenant_id, user_id, session_id], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _session_dir(self, summary: SessionSummary, create: bool = False) -> Path:
        date_dir = self.base_dir / summary.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d")
        scope = self._scope_digest(summary.tenant_id, summary.user_id, summary.session_id)
        directory = date_dir / f"{self._safe_component(summary.session_id, 'session')}-{scope}"
        resolved = directory.resolve()
        if self.base_dir not in resolved.parents:
            raise ValueError("Unsafe session summary path")
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(resolved, 0o700)
            except OSError:
                pass
        return resolved

    def _get_summary_path(self, summary: SessionSummary, suffix: str = ".md") -> Path:
        """为每个回合/总结产生唯一路径，不覆盖同 session 旧结果。"""
        if suffix not in {".md", ".json"}:
            raise ValueError("Unsupported summary suffix")
        timestamp = summary.timestamp.astimezone(timezone.utc).strftime("%H%M%S_%f")
        turn = self._safe_component(summary.turn_id or "turn", "turn")
        summary_id = self._safe_component(summary.summary_id, "summary")
        return self._session_dir(summary, create=True) / f"{timestamp}_{turn}_{summary_id}{suffix}"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(path.parent),
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_name = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def save_summary(self, summary: SessionSummary, *, consent: bool = False) -> Path:
        """持久化含完整问答的会话总结；默认拒绝，需显式用户同意。"""
        if consent is not True:
            raise PermissionError("saving a session summary requires explicit consent")
        json_path = self._get_summary_path(summary, ".json")
        markdown_path = json_path.with_suffix(".md")
        json_content = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        with self._lock:
            # JSON 先写；即使 Markdown 写入中断，仍有可回读权威记录。
            self._atomic_write(json_path, json_content + "\n")
            self._atomic_write(markdown_path, summary.to_markdown())
        logger.info("Saved consented session summary {}", summary.summary_id)
        return markdown_path

    def _matching_json_paths(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
    ) -> List[Path]:
        scope = self._scope_digest(tenant_id, user_id, session_id)
        matches = [
            path for path in self.base_dir.rglob("*.json")
            if path.parent.name.endswith(f"-{scope}")
        ]
        return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)

    def load_summary(
        self,
        session_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        turn_id: Optional[str] = None,
        summary_id: Optional[str] = None,
    ) -> Optional[SessionSummary]:
        """加载匹配作用域的最新总结，可按回合或总结 ID 进一步限定。"""
        for path in self._matching_json_paths(session_id, tenant_id, user_id):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                summary = SessionSummary.from_dict(payload)
                if summary.session_id != session_id or summary.tenant_id != tenant_id or summary.user_id != user_id:
                    continue
                if turn_id is not None and summary.turn_id != turn_id:
                    continue
                if summary_id is not None and summary.summary_id != summary_id:
                    continue
                return summary
            except Exception as exc:
                logger.warning(
                    "Ignoring invalid summary file {}: {}",
                    path.name,
                    type(exc).__name__,
                )

        # 兼容只剩 Markdown 副本的情况。
        scope = self._scope_digest(tenant_id, user_id, session_id)
        markdown_paths = sorted(
            (path for path in self.base_dir.rglob("*.md") if path.parent.name.endswith(f"-{scope}")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in markdown_paths:
            try:
                summary = SessionSummary.from_markdown(path.read_text(encoding="utf-8"))
                if turn_id is not None and summary.turn_id != turn_id:
                    continue
                if summary_id is not None and summary.summary_id != summary_id:
                    continue
                return summary
            except Exception:
                continue
        return None

    def list_summaries(
        self,
        session_id: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> List[SessionSummary]:
        summaries: List[SessionSummary] = []
        for path in self._matching_json_paths(session_id, tenant_id, user_id):
            try:
                summaries.append(SessionSummary.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return summaries

    def search_similar_sessions(
        self,
        query: str,
        limit: int = 5,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        current_session_id: Optional[str] = None,
    ) -> List[Path]:
        """本地词项重叠检索，明确不声称是向量语义搜索。"""
        query_tokens = set(re.findall(r"[\w\u3400-\u9fff]+", query.lower()))
        ranked: List[tuple[float, float, Path]] = []
        for path in self.base_dir.rglob("*.json"):
            try:
                summary = SessionSummary.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if summary.tenant_id != tenant_id or summary.user_id != user_id:
                continue
            if current_session_id is not None and summary.session_id == current_session_id:
                continue
            document_tokens = set(re.findall(
                r"[\w\u3400-\u9fff]+", f"{summary.question} {summary.final_answer}".lower()
            ))
            score = len(query_tokens & document_tokens) / len(query_tokens | document_tokens) if query_tokens | document_tokens else 0.0
            ranked.append((score, path.stat().st_mtime, path.with_suffix(".md")))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [path for _, _, path in ranked[:max(0, limit)]]

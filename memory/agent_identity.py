"""Agent 能力与协作经验的可逆、原子持久化。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Mapping, Optional
import base64
import hashlib
import json
import os
import re
import tempfile

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:  # Unix/macOS 上用于跨进程 read-modify-write 互斥。
    import fcntl
except ImportError:  # pragma: no cover - Windows 上仍有线程锁+原子 replace。
    fcntl = None  # type: ignore


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


@dataclass
class CollaborationRecord:
    partner_agent: str
    collaboration_count: int
    efficiency_improvement: float
    notes: str = ""


@dataclass
class ToolUsageStats:
    tool_name: str
    usage_count: int
    success_rate: float
    avg_execution_time: float = 0.0
    # 新版保留原始累计量，避免用 EMA 冒充实际成功率/均值。
    success_count: Optional[int] = None
    total_execution_time: Optional[float] = None


@dataclass
class AgentIdentity:
    agent_id: str
    agent_type: str
    core_capabilities: List[str]
    expertise_domains: List[str]
    collaboration_records: List[CollaborationRecord] = field(default_factory=list)
    tool_usage_stats: List[ToolUsageStats] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    last_updated: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    revision: int = 1
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(self.agent_type, str) or not self.agent_type.strip():
            raise ValueError("agent_type must be a non-empty string")
        self.created_at = _parse_datetime(self.created_at)
        self.last_updated = _parse_datetime(self.last_updated)
        self.core_capabilities = [str(item) for item in self.core_capabilities]
        self.expertise_domains = [str(item) for item in self.expertise_domains]
        self.metadata = dict(self.metadata)

    def update_collaboration(
        self,
        partner_agent: str,
        efficiency_improvement: float,
        notes: Optional[str] = None,
    ) -> None:
        if not partner_agent:
            raise ValueError("partner_agent must be non-empty")
        improvement = float(efficiency_improvement)
        for record in self.collaboration_records:
            if record.partner_agent == partner_agent:
                previous_count = max(0, int(record.collaboration_count))
                record.efficiency_improvement = (
                    (record.efficiency_improvement * previous_count + improvement)
                    / (previous_count + 1)
                )
                record.collaboration_count = previous_count + 1
                if notes is not None:
                    record.notes = notes
                self.last_updated = _utcnow()
                self.revision += 1
                return
        self.collaboration_records.append(CollaborationRecord(
            partner_agent=partner_agent,
            collaboration_count=1,
            efficiency_improvement=improvement,
            notes=notes or "",
        ))
        self.last_updated = _utcnow()
        self.revision += 1

    def update_tool_stats(self, tool_name: str, success: bool, execution_time: float) -> None:
        if not tool_name:
            raise ValueError("tool_name must be non-empty")
        duration = max(0.0, float(execution_time))
        for stats in self.tool_usage_stats:
            if stats.tool_name == tool_name:
                previous_count = max(0, int(stats.usage_count))
                successes = (
                    stats.success_count
                    if stats.success_count is not None
                    else int(round(float(stats.success_rate) * previous_count))
                )
                total_time = (
                    stats.total_execution_time
                    if stats.total_execution_time is not None
                    else float(stats.avg_execution_time) * previous_count
                )
                stats.usage_count = previous_count + 1
                stats.success_count = max(0, int(successes)) + (1 if success else 0)
                stats.total_execution_time = max(0.0, float(total_time)) + duration
                stats.success_rate = stats.success_count / stats.usage_count
                stats.avg_execution_time = stats.total_execution_time / stats.usage_count
                self.last_updated = _utcnow()
                self.revision += 1
                return
        self.tool_usage_stats.append(ToolUsageStats(
            tool_name=tool_name,
            usage_count=1,
            success_rate=1.0 if success else 0.0,
            avg_execution_time=duration,
            success_count=1 if success else 0,
            total_execution_time=duration,
        ))
        self.last_updated = _utcnow()
        self.revision += 1

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["last_updated"] = self.last_updated.isoformat()
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentIdentity":
        return cls(
            agent_id=str(data["agent_id"]),
            agent_type=str(data.get("agent_type") or data["agent_id"]),
            core_capabilities=list(data.get("core_capabilities") or []),
            expertise_domains=list(data.get("expertise_domains") or []),
            collaboration_records=[
                CollaborationRecord(**dict(item)) for item in data.get("collaboration_records", [])
            ],
            tool_usage_stats=[ToolUsageStats(**dict(item)) for item in data.get("tool_usage_stats", [])],
            created_at=_parse_datetime(data.get("created_at")),
            last_updated=_parse_datetime(data.get("last_updated")),
            metadata=dict(data.get("metadata") or {}),
            revision=int(data.get("revision", 1)),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_markdown(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        lines = [
            f"# Agent: {self.agent_id}",
            "",
            f"**Agent 类型**: {self.agent_type}",
            f"**修订版本**: {self.revision}",
            "",
            "## 核心能力",
            *[f"- {capability}" for capability in self.core_capabilities],
            "",
            "## 专长领域",
            *[f"- {domain}" for domain in self.expertise_domains],
            "",
            "## 协作经验",
            "",
        ]
        for record in self.collaboration_records:
            lines.append(
                f"- 与 {record.partner_agent} 协作 {record.collaboration_count} 次，"
                f"平均效率变化 {record.efficiency_improvement:.1%}"
            )
            if record.notes:
                lines.append(f"  > {record.notes}")
        lines.extend(["", "## 工具使用统计", ""])
        for stats in self.tool_usage_stats:
            lines.append(
                f"- {stats.tool_name}: {stats.usage_count} 次（成功率 {stats.success_rate:.1%}，"
                f"平均耗时 {stats.avg_execution_time:.3f}s）"
            )
        lines.extend([
            "",
            f"**创建时间**: {self.created_at.isoformat()}",
            f"**最后更新**: {self.last_updated.isoformat()}",
            "",
            "<!-- MEDIX_AGENT_IDENTITY_V2",
            encoded,
            "-->",
            "",
        ])
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, agent_id: str, markdown_content: str) -> "AgentIdentity":
        match = re.search(
            r"<!--\s*MEDIX_AGENT_IDENTITY_V2\s*\n([A-Za-z0-9+/=\s]+?)\n-->",
            markdown_content,
            flags=re.MULTILINE,
        )
        if match:
            encoded = re.sub(r"\s+", "", match.group(1))
            try:
                payload = json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid AgentIdentity payload") from exc
            identity = cls.from_dict(payload)
            if identity.agent_id != agent_id:
                raise ValueError("AgentIdentity payload does not match requested agent_id")
            return identity

        # 旧版 Markdown 的有限迁移解析；不伪造旧文件里没有的统计。
        type_match = re.search(r"\*\*Agent 类型\*\*:\s*(.+)", markdown_content)
        agent_type = type_match.group(1).strip() if type_match else (
            agent_id.split("_", 1)[0] if "_" in agent_id else agent_id
        )

        def section_items(title: str) -> List[str]:
            section = re.search(
                rf"^## {re.escape(title)}\s*$\n(.*?)(?=^## |^\*\*|\Z)",
                markdown_content,
                flags=re.MULTILINE | re.DOTALL,
            )
            if not section:
                return []
            return [
                line[2:].strip() for line in section.group(1).splitlines()
                if line.startswith("- ")
            ]

        return cls(
            agent_id=agent_id,
            agent_type=agent_type,
            core_capabilities=section_items("核心能力"),
            expertise_domains=section_items("专长领域"),
            metadata={"migrated_from_legacy_markdown": True},
        )


class AgentIdentityManager:
    """AgentIdentity 的路径安全、原子读写管理器。"""

    def __init__(self, base_dir: str = "memory/agents", default_tenant_id: str = "default") -> None:
        if not default_tenant_id:
            raise ValueError("default_tenant_id must be non-empty")
        requested = Path(base_dir).expanduser()
        self.base_dir = (requested if requested.is_absolute() else PROJECT_ROOT / requested).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.base_dir, 0o700)
        except OSError:
            pass
        self.default_tenant_id = default_tenant_id
        self._lock = RLock()

    @staticmethod
    def _safe_component(value: str, prefix: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix} identifier must be non-empty")
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:48] or prefix
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return f"{cleaned}-{digest}"

    def _identity_dir(self, agent_id: str, tenant_id: Optional[str], create: bool = False) -> Path:
        tenant = tenant_id or self.default_tenant_id
        path = self.base_dir / self._safe_component(tenant, "tenant") / self._safe_component(agent_id, "agent")
        resolved = path.resolve()
        if self.base_dir not in resolved.parents:
            raise ValueError("Unsafe AgentIdentity path")
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(resolved, 0o700)
            except OSError:
                pass
        return resolved

    def _get_identity_path(self, agent_id: str, tenant_id: Optional[str] = None) -> Path:
        return self._identity_dir(agent_id, tenant_id, create=True) / "IDENTITY.md"

    @contextmanager
    def _file_lock(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / ".identity.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
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

    def _load_unlocked(self, agent_id: str, tenant_id: Optional[str]) -> Optional[AgentIdentity]:
        directory = self._identity_dir(agent_id, tenant_id, create=False)
        json_path = directory / "IDENTITY.json"
        markdown_path = directory / "IDENTITY.md"
        try:
            if json_path.exists():
                identity = AgentIdentity.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
                if identity.agent_id != agent_id:
                    raise ValueError("Stored identity does not match requested agent_id")
                return identity
            if markdown_path.exists():
                return AgentIdentity.from_markdown(agent_id, markdown_path.read_text(encoding="utf-8"))

            # 只对无路径分隔符的旧 agent_id 尝试旧目录。
            if re.fullmatch(r"[A-Za-z0-9_.-]+", agent_id):
                legacy = (self.base_dir / agent_id / "IDENTITY.md").resolve()
                if self.base_dir in legacy.parents and legacy.exists():
                    return AgentIdentity.from_markdown(agent_id, legacy.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Error loading agent identity: {}", type(exc).__name__)
        return None

    def load_identity(self, agent_id: str, *, tenant_id: Optional[str] = None) -> Optional[AgentIdentity]:
        directory = self._identity_dir(agent_id, tenant_id, create=True)
        with self._lock, self._file_lock(directory):
            return self._load_unlocked(agent_id, tenant_id)

    def _save_unlocked(self, identity: AgentIdentity, tenant_id: Optional[str]) -> Path:
        directory = self._identity_dir(identity.agent_id, tenant_id, create=True)
        json_path = directory / "IDENTITY.json"
        markdown_path = directory / "IDENTITY.md"
        self._atomic_write(
            json_path,
            json.dumps(identity.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        self._atomic_write(markdown_path, identity.to_markdown())
        return markdown_path

    def save_identity(self, identity: AgentIdentity, *, tenant_id: Optional[str] = None) -> Path:
        directory = self._identity_dir(identity.agent_id, tenant_id, create=True)
        with self._lock, self._file_lock(directory):
            return self._save_unlocked(identity, tenant_id)

    def create_identity(
        self,
        agent_id: str,
        agent_type: str,
        core_capabilities: List[str],
        expertise_domains: List[str],
        *,
        tenant_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AgentIdentity:
        identity = AgentIdentity(
            agent_id=agent_id,
            agent_type=agent_type,
            core_capabilities=core_capabilities,
            expertise_domains=expertise_domains,
            metadata=dict(metadata or {}),
        )
        self.save_identity(identity, tenant_id=tenant_id)
        return identity

    def update_collaboration(
        self,
        agent_id: str,
        partner_agent: str,
        efficiency_improvement: float,
        *,
        tenant_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[AgentIdentity]:
        directory = self._identity_dir(agent_id, tenant_id, create=True)
        with self._lock, self._file_lock(directory):
            identity = self._load_unlocked(agent_id, tenant_id)
            if identity is None:
                return None
            identity.update_collaboration(partner_agent, efficiency_improvement, notes)
            self._save_unlocked(identity, tenant_id)
            return identity

    def update_tool_stats(
        self,
        agent_id: str,
        tool_name: str,
        success: bool,
        execution_time: float,
        *,
        tenant_id: Optional[str] = None,
    ) -> Optional[AgentIdentity]:
        directory = self._identity_dir(agent_id, tenant_id, create=True)
        with self._lock, self._file_lock(directory):
            identity = self._load_unlocked(agent_id, tenant_id)
            if identity is None:
                return None
            identity.update_tool_stats(tool_name, success, execution_time)
            self._save_unlocked(identity, tenant_id)
            return identity

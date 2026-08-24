"""Request-scoped blackboard for parallel Swarm work."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional
import uuid

from core.conversation_context import RequestContext
from .events import Event, EventType


class TaskStatus(Enum):
    PENDING = "pending"
    CLAIMED = "claimed"  # legacy value
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    id: str
    type: str
    description: str
    assigned_agent: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)

    @property
    def assigned_to(self) -> str:
        """Compatibility alias used by an older timeout logger."""
        return self.assigned_agent

    def can_be_executed(self) -> bool:
        return self.status == TaskStatus.PENDING

    def start(self) -> None:
        if not self.can_be_executed():
            raise ValueError(
                f"SubTask {self.id} cannot be started (status={self.status.value})"
            )
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def complete(self, result: Dict[str, Any]) -> None:
        if self.status not in (TaskStatus.IN_PROGRESS, TaskStatus.CLAIMED):
            raise ValueError(
                f"SubTask {self.id} cannot complete from {self.status.value}"
            )
        self.result = result
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()

    def fail(self, error: str) -> None:
        if self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return
        self.result = {"error": str(error)}
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()


@dataclass(frozen=True)
class Contribution:
    agent_id: str
    subtask_id: str
    result: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedContext:
    """A blackboard owned by one turn, never attached to reusable Agents."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        request_context: Optional[RequestContext] = None,
    ):
        self.session_id = session_id or (
            request_context.session_id if request_context else str(uuid.uuid4())
        )
        self.request_context = request_context
        self.created_at = datetime.now()
        self.data: Dict[str, Any] = {}
        self.events: List[Event] = []
        self.task_decomposition: Dict[str, SubTask] = {}
        self.agent_contributions: Dict[str, List[Contribution]] = defaultdict(list)
        self.memory_pool: Dict[str, Any] = {}
        self._lock = RLock()

    def publish_event(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
    ) -> List[Event]:
        with self._lock:
            events = list(self.events)
        if event_type:
            events = [event for event in events if event.type == event_type]
        if source_agent:
            events = [event for event in events if event.source_agent == source_agent]
        if target_agent:
            events = [event for event in events if event.is_for_agent(target_agent)]
        return events

    def add_subtask(self, subtask: SubTask) -> None:
        with self._lock:
            if subtask.id in self.task_decomposition:
                raise ValueError(f"Duplicate SubTask id: {subtask.id}")
            self.task_decomposition[subtask.id] = subtask
            self.publish_event(Event(
                type=EventType.TASK_DECOMPOSED,
                source_agent="lead_agent",
                data={
                    "subtask_id": subtask.id,
                    "type": subtask.type,
                    "assigned_agent": subtask.assigned_agent,
                    "dependencies": list(subtask.dependencies),
                },
            ))

    def get_subtask(self, subtask_id: str) -> Optional[SubTask]:
        with self._lock:
            return self.task_decomposition.get(subtask_id)

    def _dependencies_completed(self, subtask: SubTask) -> bool:
        return all(
            dependency in self.task_decomposition
            and self.task_decomposition[dependency].status == TaskStatus.COMPLETED
            for dependency in subtask.dependencies
        )

    def _dependency_failed(self, subtask: SubTask) -> bool:
        return any(
            dependency not in self.task_decomposition
            or self.task_decomposition[dependency].status == TaskStatus.FAILED
            for dependency in subtask.dependencies
        )

    def get_subtasks_for_agent(self, agent_id: str) -> List[SubTask]:
        """Return pending tasks whose declared dependencies are complete."""
        with self._lock:
            return [
                subtask
                for subtask in self.task_decomposition.values()
                if subtask.assigned_agent == agent_id
                and subtask.can_be_executed()
                and self._dependencies_completed(subtask)
            ]

    def get_ready_subtasks(self) -> List[SubTask]:
        with self._lock:
            return [
                subtask
                for subtask in self.task_decomposition.values()
                if subtask.can_be_executed() and self._dependencies_completed(subtask)
            ]

    def fail_blocked_subtasks(self) -> int:
        """Fail tasks whose dependencies failed or do not exist."""
        failed = []
        with self._lock:
            for subtask in self.task_decomposition.values():
                if subtask.can_be_executed() and self._dependency_failed(subtask):
                    subtask.fail("dependency_failed_or_missing")
                    failed.append(subtask)
            for subtask in failed:
                self.publish_event(Event(
                    type=EventType.SUBTASK_FAILED,
                    source_agent="swarm_coordinator",
                    data={"subtask_id": subtask.id, "error": "dependency_failed_or_missing"},
                ))
        return len(failed)

    def start_subtask(self, subtask_id: str) -> bool:
        """Atomically claim/start a ready subtask."""
        with self._lock:
            subtask = self.task_decomposition.get(subtask_id)
            if (
                not subtask
                or not subtask.can_be_executed()
                or not self._dependencies_completed(subtask)
            ):
                return False
            subtask.start()
            self.publish_event(Event(
                type=EventType.SUBTASK_STARTED,
                source_agent=subtask.assigned_agent,
                data={"subtask_id": subtask_id},
            ))
            return True

    def complete_subtask(
        self,
        subtask_id: str,
        agent_id: str,
        result: Dict[str, Any],
        confidence: float = 1.0,
    ) -> None:
        with self._lock:
            subtask = self.task_decomposition.get(subtask_id)
            if not subtask:
                raise ValueError(f"SubTask {subtask_id} not found")
            if subtask.assigned_agent != agent_id:
                raise ValueError(f"SubTask {subtask_id} not assigned to {agent_id}")
            subtask.complete(result)
            contribution = Contribution(
                agent_id=agent_id,
                subtask_id=subtask_id,
                result=result,
                confidence=max(0.0, min(1.0, float(confidence))),
            )
            self.agent_contributions[agent_id].append(contribution)
            self.publish_event(Event(
                type=EventType.SUBTASK_COMPLETED,
                source_agent=agent_id,
                data={
                    "subtask_id": subtask_id,
                    "result_summary": str(result)[:200],
                },
            ))

    def fail_subtask(self, subtask_id: str, agent_id: str, error: str) -> None:
        with self._lock:
            subtask = self.task_decomposition.get(subtask_id)
            if not subtask:
                return
            subtask.fail(error)
            self.publish_event(Event(
                type=EventType.SUBTASK_FAILED,
                source_agent=agent_id,
                data={"subtask_id": subtask_id, "error": str(error)[:300]},
            ))

    def cancel_unfinished(self, reason: str = "timeout") -> List[str]:
        cancelled = []
        with self._lock:
            for subtask in self.task_decomposition.values():
                if subtask.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    subtask.fail(reason)
                    cancelled.append(subtask.id)
                    self.publish_event(Event(
                        type=EventType.SUBTASK_FAILED,
                        source_agent="swarm_coordinator",
                        data={"subtask_id": subtask.id, "error": reason},
                    ))
        return cancelled

    def get_dependency_results(self, subtask_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            subtask = self.task_decomposition.get(subtask_id)
            if not subtask:
                return []
            results = []
            for dependency_id in subtask.dependencies:
                dependency = self.task_decomposition.get(dependency_id)
                if dependency and dependency.status == TaskStatus.COMPLETED:
                    results.append({
                        "subtask_id": dependency.id,
                        "type": dependency.type,
                        "agent_id": dependency.assigned_agent,
                        "result": dependency.result,
                    })
            return results

    def get_contributions(
        self,
        agent_id: Optional[str] = None,
        subtask_id: Optional[str] = None,
    ) -> List[Contribution]:
        with self._lock:
            if agent_id:
                contributions = list(self.agent_contributions.get(agent_id, ()))
            else:
                contributions = [
                    contribution
                    for values in self.agent_contributions.values()
                    for contribution in values
                ]
        if subtask_id:
            contributions = [
                contribution
                for contribution in contributions
                if contribution.subtask_id == subtask_id
            ]
        return contributions

    def get_all_completed_subtasks(self) -> List[SubTask]:
        with self._lock:
            return [
                subtask
                for subtask in self.task_decomposition.values()
                if subtask.status == TaskStatus.COMPLETED
            ]

    def get_incomplete_subtasks(self) -> List[SubTask]:
        with self._lock:
            return [
                subtask
                for subtask in self.task_decomposition.values()
                if subtask.status != TaskStatus.COMPLETED
            ]

    def is_all_subtasks_completed(self) -> bool:
        with self._lock:
            return bool(self.task_decomposition) and all(
                subtask.status == TaskStatus.COMPLETED
                for subtask in self.task_decomposition.values()
            )

    def is_all_subtasks_finished(self) -> bool:
        with self._lock:
            terminal = (TaskStatus.COMPLETED, TaskStatus.FAILED)
            return bool(self.task_decomposition) and all(
                subtask.status in terminal
                for subtask in self.task_decomposition.values()
            )

    def set_data(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
            self.publish_event(Event(
                type=EventType.CONTEXT_UPDATED,
                source_agent="system",
                data={"key": key},
            ))

    def get_data(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.data.get(key, default)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            status_counts = {
                status.value: sum(
                    1 for task in self.task_decomposition.values() if task.status == status
                )
                for status in TaskStatus
            }
            return {
                "session_id": self.session_id,
                "turn_id": self.request_context.turn_id if self.request_context else None,
                "created_at": self.created_at.isoformat(),
                "total_events": len(self.events),
                "total_subtasks": len(self.task_decomposition),
                "completed_subtasks": status_counts[TaskStatus.COMPLETED.value],
                "failed_subtasks": status_counts[TaskStatus.FAILED.value],
                "status_counts": status_counts,
                "agent_count": len(self.agent_contributions),
                "agents": list(self.agent_contributions.keys()),
            }


__all__ = ["SharedContext", "SubTask", "Contribution", "TaskStatus"]

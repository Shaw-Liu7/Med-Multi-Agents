"""Conversation boundary and request-scoped Swarm coordinator."""
from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple
from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMClient
from core.conversation_context import (
    RequestContext,
    complete_turns_from_messages,
    memories_from_results,
)
from core.safety_hooks import finalize_output, precheck_input
from memory import LongTermMemory, SessionSummary, SessionSummaryManager, ShortTermMemory

from .events import Event, EventType
from .lead_agent import LeadAgent
from .shared_context import SharedContext, SubTask, TaskStatus


try:
    from constraints import ConstraintValidator, SafetyGate
except ImportError:
    ConstraintValidator = None
    SafetyGate = None


class SwarmCoordinator:
    """Own routing, canonical transcript writes and final safety enforcement.

    Reuse one coordinator for many turns. Agents are reusable, but all mutable
    request state (tool counts, SharedContext and prompts) remains per-run.
    Calls for one conversation scope are serialized; unrelated sessions remain
    concurrent.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True,
        *,
        short_term_memory: Optional[ShortTermMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        session_manager: Optional[SessionSummaryManager] = None,
        swarm_timeout_seconds: Optional[float] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm

        self.lead_agent = LeadAgent(llm_client=self.llm_client)
        self.consultation_agent = ConsultationAgent(llm_client=self.llm_client)
        self.diagnostic_agent = DiagnosticAgent(llm_client=self.llm_client)
        self.research_agent = ResearchAgent(llm_client=self.llm_client)
        self.worker_pool: List[Any] = [
            self.consultation_agent,
            self.diagnostic_agent,
            self.research_agent,
        ]

        self.session_manager = session_manager or SessionSummaryManager()
        self.short_term_memory = short_term_memory or ShortTermMemory(storage_type="memory")
        self.long_term_memory = long_term_memory or LongTermMemory()
        for worker in self.worker_pool:
            worker.loop.short_term_memory = self.short_term_memory
            worker.loop.long_term_memory = self.long_term_memory
        self.validator = ConstraintValidator() if ConstraintValidator else None
        limits = (
            self.validator.get_swarm_limits()
            if self.validator and hasattr(self.validator, "get_swarm_limits")
            else {
                "max_parallel_tasks": 5,
                "timeout_seconds": 90,
            }
        )
        policy_timeout = max(1.0, float(limits["timeout_seconds"]))
        requested_timeout = (
            policy_timeout
            if swarm_timeout_seconds is None
            else max(1.0, float(swarm_timeout_seconds))
        )
        self.swarm_timeout_seconds = min(requested_timeout, policy_timeout)
        self.max_parallel_tasks = max(1, int(limits["max_parallel_tasks"]))
        self.safety_gate = self._create_safety_gate()
        self._session_locks: Dict[Tuple[str, str, str], asyncio.Lock] = {}

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")

    def _create_safety_gate(self) -> Any:
        if not SafetyGate:
            return None
        last_error: Optional[Exception] = None
        for args in ((self.validator,), ()):
            try:
                return SafetyGate(*args)
            except TypeError:
                continue
            except Exception as exc:
                last_error = exc
                break
        error_type = type(last_error).__name__ if last_error else "TypeError"
        logger.error(
            f"Failed to initialize coordinator SafetyGate: error_type={error_type}"
        )
        raise RuntimeError("medical_safety_gate_initialization_failed") from last_error

    def _get_agent_by_id(self, agent_id: str) -> Optional[Any]:
        return {
            "consultation_agent": self.consultation_agent,
            "diagnostic_agent": self.diagnostic_agent,
            "research_agent": self.research_agent,
        }.get(agent_id)

    async def process(
        self,
        question: str,
        context: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        turn_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process one canonical user turn (legacy positional API preserved)."""
        start_time = datetime.now()
        resolved_session = session_id or (
            f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        )

        if isinstance(context, RequestContext):
            base_context = context
            if base_context.raw_question != str(question).strip():
                base_context = replace(base_context, raw_question=str(question).strip())
        else:
            base_context = RequestContext.from_legacy(
                str(question),
                session_id=resolved_session,
                context=context,
                tenant_id=tenant_id,
                user_id=user_id,
                turn_id=turn_id,
            )

        scope = (
            base_context.tenant_id,
            base_context.user_id,
            base_context.session_id,
        )
        lock = self._session_locks.setdefault(scope, asyncio.Lock())
        async with lock:
            return await self._process_locked(base_context, start_time)

    async def _process_locked(
        self,
        base_context: RequestContext,
        start_time: datetime,
    ) -> Dict[str, Any]:
        logger.info("Processing a scoped conversation turn")
        request_context = await self._hydrate_context(base_context)

        blocked_response = await precheck_input(
            validator=self.validator,
            safety_gate=self.safety_gate,
            agent_id="final_answer",
            question=request_context.raw_question,
            request_context=request_context,
        )

        shared_for_summary: Optional[SharedContext] = None
        mode = "safety_short_circuit"
        if blocked_response:
            result: Dict[str, Any] = {
                "answer": blocked_response,
                "swarm_enabled": False,
                "suggestions": ["如有紧急症状，请立即联系急救服务或前往急诊"],
                "disclaimer": "紧急风险提示不能替代现场医疗评估。",
                "warning": "input_blocked_by_safety_gate",
            }
            subtasks_count = 0
        else:
            assessment = await self.lead_agent.assess_and_decompose(
                request_context,
                request_context=request_context,
            )
            subtasks = assessment.get("subtasks", [])
            plan_allowed = self._validate_decomposition(
                request_context.raw_question,
                subtasks,
            )
            if not plan_allowed:
                # Never execute an invalid/blocked LLM plan. A single bounded
                # Consultation fallback still passes the same final safety gate.
                subtasks = []
                assessment = {
                    "subtasks": [],
                    "reason": "task_decomposition_blocked_by_policy",
                }
            subtasks_count = len(subtasks)

            if len(subtasks) == 1:
                mode = "single_agent"
                result = await self._process_single(request_context, subtasks[0])
            elif len(subtasks) >= 2 and self.enable_swarm:
                mode = "swarm"
                result = await self._process_with_swarm(
                    request_context=request_context,
                    assessment=assessment,
                    start_time=start_time,
                )
                shared_for_summary = result.pop("_shared_context", None)
            else:
                mode = "fallback" if not subtasks else "disabled_swarm"
                fallback_context = request_context.for_task(
                    "直接回答当前健康问题；信息不足时明确需要补充哪些信息。"
                )
                result = await self.consultation_agent.process(
                    self._agent_input(fallback_context),
                    request_context=fallback_context,
                )
                result.update({"swarm_enabled": False})

        final_answer = await finalize_output(
            validator=self.validator,
            safety_gate=self.safety_gate,
            agent_id="final_answer",
            question=request_context.raw_question,
            output=str(result.get("answer", "")),
            request_context=request_context,
        )
        result["answer"] = final_answer
        result.setdefault("suggestions", [])
        result.setdefault(
            "disclaimer",
            "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。",
        )
        result.update({
            "session_id": request_context.session_id,
            "turn_id": request_context.turn_id,
            "tenant_id": request_context.tenant_id,
            "input_redacted": request_context.input_redacted,
            "swarm_enabled": bool(result.get("swarm_enabled", False)),
        })

        end_time = datetime.now()
        await self._persist_canonical_turn(
            request_context,
            final_answer,
            mode=mode,
            subtasks_count=subtasks_count,
            elapsed=(end_time - start_time).total_seconds(),
            timeout_occurred=bool(result.get("timeout_occurred", False)),
        )
        if shared_for_summary is not None and request_context.long_term_memory_consent:
            await self._save_local_swarm_summary(
                request_context,
                shared_for_summary,
                final_answer,
                start_time,
                end_time,
            )
        return result

    async def _hydrate_context(self, base: RequestContext) -> RequestContext:
        history_messages: List[Mapping[str, Any]] = []
        rolling_summary = base.rolling_summary
        try:
            history_messages = await asyncio.to_thread(
                self.short_term_memory.get_recent_messages,
                base.session_id,
                11,
                tenant_id=base.tenant_id,
                user_id=base.user_id,
                char_budget=10_000,
                turn_limit=5,
            )
            session = await asyncio.to_thread(
                self.short_term_memory.get_session,
                base.session_id,
                tenant_id=base.tenant_id,
                user_id=base.user_id,
            )
            if session is not None:
                rolling_summary = str(getattr(session, "rolling_summary", "") or rolling_summary)
            if not rolling_summary:
                summary_message = next(
                    (
                        item for item in history_messages
                        if str(item.get("role", "")) == "system"
                    ),
                    None,
                )
                if summary_message:
                    rolling_summary = str(summary_message.get("content", ""))
        except Exception as exc:
            logger.error(
                f"Failed to load short-term context: error_type={type(exc).__name__}"
            )

        memory_results: List[Mapping[str, Any]] = []
        if base.memory_consent and getattr(self.long_term_memory, "enabled", False):
            try:
                search = self.long_term_memory.search_similar_sessions
                search_kwargs: Dict[str, Any] = {
                    "tenant_id": base.tenant_id,
                    "user_id": base.user_id,
                    "min_score": 0.55,
                    "current_session_id": base.session_id,
                }
                try:
                    if "consent" in inspect.signature(search).parameters:
                        search_kwargs["consent"] = True
                except (TypeError, ValueError):
                    pass
                memory_results = await asyncio.to_thread(
                    search,
                    base.raw_question,
                    3,
                    **search_kwargs,
                )
            except Exception as exc:
                logger.error(
                    f"Failed to retrieve long-term context: "
                    f"error_type={type(exc).__name__}"
                )

        stored_turns = complete_turns_from_messages(history_messages, limit=5)
        recent_turns = tuple(base.recent_turns) + tuple(stored_turns)
        # De-duplicate when a caller supplied the same transcript explicitly.
        deduplicated = []
        seen = set()
        for turn in recent_turns:
            fingerprint = (turn.user, turn.assistant)
            if fingerprint not in seen:
                deduplicated.append(turn)
                seen.add(fingerprint)

        memories = memories_from_results(
            tuple(base.retrieved_memories) + tuple(memory_results),
            limit=3,
            min_score=0.55,
        )
        return replace(
            base,
            rolling_summary=rolling_summary,
            recent_turns=tuple(deduplicated[-5:]),
            retrieved_memories=memories,
        )

    def _validate_decomposition(self, question: str, subtasks: List[Any]) -> bool:
        if not self.validator or not hasattr(self.validator, "validate_task_decomposition"):
            return True
        try:
            validation = self.validator.validate_task_decomposition(question, subtasks)
            if isinstance(validation, Mapping) and not validation.get("valid", True):
                logger.warning(f"Task decomposition warning: {validation}")
                return not bool(validation.get("blocked", True))
            return True
        except Exception as exc:
            logger.error(
                f"Task decomposition validation failed: "
                f"error_type={type(exc).__name__}"
            )
            return False

    def _validate_swarm_result(
        self,
        shared_context: SharedContext,
        final_answer: str,
        question: str,
    ) -> bool:
        if not self.validator or not hasattr(self.validator, "validate_swarm_result"):
            return True
        try:
            validation = self.validator.validate_swarm_result(
                shared_context.get_contributions(),
                final_answer,
                user_input=question,
            )
            if isinstance(validation, Mapping):
                if validation.get("blocked") or not validation.get("valid", True):
                    logger.warning("Swarm quality gate blocked the synthesized result")
                    return False
            return True
        except Exception as exc:
            logger.error(
                f"Swarm quality validation failed: error_type={type(exc).__name__}"
            )
            return False

    async def _process_single(
        self,
        context: RequestContext,
        task: Mapping[str, Any],
    ) -> Dict[str, Any]:
        requested_agent_id = str(task.get("assigned_agent", "consultation_agent"))
        agent = self._get_agent_by_id(requested_agent_id) or self.consultation_agent
        task_context = context.for_task(
            str(task.get("description", "直接回答当前用户问题"))
        )
        result = await agent.process(
            self._agent_input(task_context),
            request_context=task_context,
        )
        await self._persist_tool_traces(
            context,
            agent.agent_id,
            result.get("tool_trace", ()),
        )
        result.update({
            "swarm_enabled": False,
            "route_reason": f"单任务路由到 {agent.agent_id}",
        })
        return result

    @staticmethod
    def _agent_input(context: RequestContext) -> Dict[str, Any]:
        return {
            "question": context.raw_question,
            "session_id": context.session_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "turn_id": context.turn_id,
        }

    async def _process_with_swarm(
        self,
        *,
        request_context: RequestContext,
        assessment: Dict[str, Any],
        start_time: datetime,
    ) -> Dict[str, Any]:
        shared_context = SharedContext(
            session_id=request_context.session_id,
            request_context=request_context,
        )
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="swarm_coordinator",
            data={
                "turn_id": request_context.turn_id,
                "num_subtasks": len(assessment.get("subtasks", [])),
            },
        ))
        subtasks = self.lead_agent.create_subtasks(assessment, shared_context)

        timeout_occurred = False
        try:
            await asyncio.wait_for(
                self._execute_ready_subtasks(shared_context, request_context),
                timeout=self.swarm_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            cancelled = shared_context.cancel_unfinished("swarm_timeout")
            logger.warning(
                f"Swarm execution timed out after {self.swarm_timeout_seconds}s; "
                f"cancelled {len(cancelled)} tasks"
            )

        final_answer = await self.lead_agent.synthesize_results(
            request_context,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred,
            request_context=request_context,
        )
        final_answer = await finalize_output(
            validator=self.validator,
            safety_gate=self.safety_gate,
            agent_id="final_answer",
            question=request_context.raw_question,
            output=final_answer,
            request_context=request_context,
        )
        swarm_quality_failed = not self._validate_swarm_result(
            shared_context,
            final_answer,
            request_context.raw_question,
        )
        if swarm_quality_failed:
            final_answer = (
                "抱歉，本次多模块分析未达到安全汇总标准，无法提供可靠的个体化结论。"
                "请补充信息后重试或咨询专业医生；如有紧急症状请立即就医。"
            )
        end_time = datetime.now()
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="swarm_coordinator",
            data={
                "duration": (end_time - start_time).total_seconds(),
                "agents_count": len(shared_context.agent_contributions),
                "timeout_occurred": timeout_occurred,
            },
        ))

        completed_agents = list(shared_context.agent_contributions.keys())
        result: Dict[str, Any] = {
            "answer": final_answer,
            "swarm_enabled": True,
            "agents_involved": completed_agents,
            "subtasks_completed": len(shared_context.get_all_completed_subtasks()),
            "subtasks_failed": sum(
                task.status == TaskStatus.FAILED for task in subtasks
            ),
            "total_time": (end_time - start_time).total_seconds(),
            "swarm_metadata": shared_context.get_summary(),
            "timeout_occurred": timeout_occurred,
            "quality_gate_failed": swarm_quality_failed,
            "suggestions": self._extract_suggestions(final_answer),
            "_shared_context": shared_context,
        }
        if timeout_occurred and not completed_agents:
            result["disclaimer"] = (
                "系统超时，未能完成分析；如有紧急症状请立即就医。"
            )
        elif timeout_occurred:
            result["disclaimer"] = (
                "以上仅基于已完成模块的部分结果，不能替代医生诊断。"
            )
        else:
            result["disclaimer"] = (
                "以上分析基于多个专业 Agent 的协作，仅供参考，不能替代医生诊断。"
            )
        return result

    async def _execute_ready_subtasks(
        self,
        shared_context: SharedContext,
        request_context: RequestContext,
    ) -> None:
        running: Dict[asyncio.Task[Any], SubTask] = {}
        try:
            while not shared_context.is_all_subtasks_finished():
                shared_context.fail_blocked_subtasks()
                available_slots = max(0, self.max_parallel_tasks - len(running))
                for subtask in shared_context.get_ready_subtasks()[:available_slots]:
                    worker = self._get_agent_by_id(subtask.assigned_agent)
                    if worker is None:
                        shared_context.fail_subtask(
                            subtask.id,
                            "swarm_coordinator",
                            f"unknown_agent:{subtask.assigned_agent}",
                        )
                        continue
                    if not shared_context.start_subtask(subtask.id):
                        continue
                    task = asyncio.create_task(self._execute_single_subtask(
                        worker,
                        subtask,
                        shared_context,
                        request_context,
                    ))
                    running[task] = subtask

                if not running:
                    if not shared_context.is_all_subtasks_finished():
                        shared_context.cancel_unfinished("unresolvable_dependencies")
                    break

                done, _ = await asyncio.wait(
                    tuple(running),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    running.pop(task, None)
                    try:
                        await task
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # _execute_single_subtask already recorded failure.
                        pass
        finally:
            pending = [task for task in running if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _execute_single_subtask(
        self,
        worker: Any,
        subtask: SubTask,
        shared_context: SharedContext,
        request_context: RequestContext,
    ) -> None:
        dependency_results = shared_context.get_dependency_results(subtask.id)
        task_context = request_context.for_task(
            subtask.description,
            collaboration_results=dependency_results,
        )
        try:
            result = await worker.process_subtask(
                subtask,
                request_context=task_context,
                shared_context=shared_context,
            )
            await self._persist_tool_traces(
                request_context,
                worker.agent_id,
                result.get("tool_trace", ()),
                subtask_id=subtask.id,
            )
            shared_context.complete_subtask(
                subtask.id,
                worker.agent_id,
                self._public_worker_result(result),
            )
            logger.info(f"{worker.agent_id}: completed {subtask.type}")
        except asyncio.CancelledError:
            shared_context.fail_subtask(subtask.id, worker.agent_id, "cancelled")
            raise
        except Exception as exc:
            shared_context.fail_subtask(
                subtask.id,
                worker.agent_id,
                "worker_execution_failed",
            )
            logger.error(
                f"Worker subtask failed: agent={worker.agent_id}, "
                f"task_type={subtask.type}, error_type={type(exc).__name__}"
            )
            raise

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext,
        request_context: Optional[RequestContext] = None,
    ) -> None:
        """Compatibility entry point; new code uses dependency-aware scheduler."""
        context = request_context or shared_context.request_context
        if context is None:
            raise ValueError("request_context is required for Worker execution")
        tasks = []
        for subtask in shared_context.get_subtasks_for_agent(worker.agent_id):
            if shared_context.start_subtask(subtask.id):
                tasks.append(asyncio.create_task(self._execute_single_subtask(
                    worker,
                    subtask,
                    shared_context,
                    context,
                )))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def _public_worker_result(result: Mapping[str, Any]) -> Dict[str, Any]:
        internal = {
            "tool_trace",
            "tool_calls_history",
            "request_context",
            "tenant_id",
            "user_id",
        }
        return {key: value for key, value in result.items() if key not in internal}

    async def _persist_tool_traces(
        self,
        context: RequestContext,
        agent_id: str,
        traces: Any,
        *,
        subtask_id: Optional[str] = None,
    ) -> None:
        for trace in traces or ():
            try:
                await asyncio.to_thread(
                    self.short_term_memory.add_message,
                    context.session_id,
                    "tool",
                    str(trace),
                    tenant_id=context.tenant_id,
                    user_id=context.user_id,
                    message_type="trace",
                    metadata={
                        "turn_id": context.turn_id,
                        "agent_id": agent_id,
                        "subtask_id": subtask_id,
                    },
                )
            except Exception as exc:
                logger.error(
                    f"Failed to store isolated tool trace: "
                    f"error_type={type(exc).__name__}"
                )

    async def _persist_canonical_turn(
        self,
        context: RequestContext,
        final_answer: str,
        *,
        mode: str,
        subtasks_count: int,
        elapsed: float,
        timeout_occurred: bool,
    ) -> None:
        metadata = {
            "turn_id": context.turn_id,
            "mode": mode,
            "subtasks_count": subtasks_count,
            "total_time": elapsed,
            "timeout_occurred": timeout_occurred,
            "input_redacted": context.input_redacted,
        }
        try:
            # One atomic boundary write: canonical raw user + final assistant.
            await asyncio.to_thread(
                self.short_term_memory.add_turn,
                context.session_id,
                context.raw_question,
                final_answer,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                metadata=metadata,
            )
        except Exception as exc:
            logger.error(
                f"Failed to persist canonical short-term turn: "
                f"error_type={type(exc).__name__}"
            )

        if not context.memory_consent:
            return
        try:
            facts = context.metadata.get("verified_facts", ())
            if not isinstance(facts, (list, tuple)):
                facts = ()
            await asyncio.to_thread(
                self.long_term_memory.add_session_summary,
                context.session_id,
                context.raw_question,
                final_answer,
                metadata,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                turn_id=context.turn_id,
                facts=facts,
                source="conversation_turn",
                confidence=0.5,
                consent=True,
                contains_phi=True,
            )
        except Exception as exc:
            logger.error(
                f"Failed to persist consented long-term memory: "
                f"error_type={type(exc).__name__}"
            )

    async def _save_local_swarm_summary(
        self,
        context: RequestContext,
        shared_context: SharedContext,
        final_answer: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        try:
            # One file per turn prevents later Swarm turns from overwriting it.
            summary = SessionSummary.from_shared_context(
                session_id=context.session_id,
                question=context.raw_question,
                shared_context=shared_context,
                final_answer=final_answer,
                start_time=start_time,
                end_time=end_time,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                turn_id=context.turn_id,
            )
            save = self.session_manager.save_summary
            save_kwargs: Dict[str, Any] = {}
            try:
                if "consent" in inspect.signature(save).parameters:
                    save_kwargs["consent"] = True
            except (TypeError, ValueError):
                pass
            await asyncio.to_thread(save, summary, **save_kwargs)
        except Exception as exc:
            logger.error(
                f"Failed to save local Swarm summary: error_type={type(exc).__name__}"
            )

    def clear_session(
        self,
        session_id: str,
        *,
        tenant_id: str = "default",
        user_id: Optional[str] = None,
    ) -> None:
        """Clear one short-term conversation scope; long-term data is retained."""
        resolved_user = user_id or f"anonymous:{session_id}"
        self.short_term_memory.clear_session(
            session_id,
            tenant_id=tenant_id,
            user_id=resolved_user,
        )
        self._session_locks.pop((tenant_id, resolved_user, session_id), None)

    async def aclose(self) -> None:
        """Release coordinator-owned ephemeral state."""
        self._session_locks.clear()
        for worker in self.worker_pool:
            if hasattr(worker, "loop"):
                worker.loop.state_manager.cleanup_old_states(hours=0)

    @staticmethod
    def _extract_suggestions(final_answer: str) -> List[str]:
        import re

        if "【核心建议】" not in final_answer:
            return []
        start = final_answer.find("【核心建议】")
        end = final_answer.find("【", start + 1)
        text = final_answer[start:] if end == -1 else final_answer[start:end]
        return re.findall(r"\d+\.\s*([^\n]+)", text)[:5]


_DEFAULT_COORDINATORS: Dict[bool, SwarmCoordinator] = {}


async def process_with_swarm(
    question: str,
    context: Optional[Mapping[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None,
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    coordinator: Optional[SwarmCoordinator] = None,
) -> Dict[str, Any]:
    """Backward-compatible convenience API with a long-lived default instance."""
    active = coordinator
    if active is None:
        active = _DEFAULT_COORDINATORS.get(enable_swarm)
        if active is None:
            active = SwarmCoordinator(enable_swarm=enable_swarm)
            _DEFAULT_COORDINATORS[enable_swarm] = active
    return await active.process(
        question,
        context,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        turn_id=turn_id,
    )


__all__ = ["SwarmCoordinator", "process_with_swarm"]

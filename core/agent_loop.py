"""Stateless-per-run LLM/tool loop.

Conversation persistence is owned by ``SwarmCoordinator``. This loop receives
an immutable request snapshot, builds a prompt once, and returns a separate
tool trace. It never writes formatted prompts or worker traces to conversation
memory.
"""
from __future__ import annotations

import asyncio
import json
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from loguru import logger

from .conversation_context import (
    RequestContext,
    memories_from_results,
    redact_text,
    thaw,
)
from .llm_client import LLMResponse
from .prompt_builder import PromptBuilder
from .safety_hooks import finalize_output, precheck_input
from .state_manager import StateManager, TaskStatus


try:
    from constraints import ConstraintValidator
except ImportError:
    ConstraintValidator = None

try:
    from constraints import SafetyGate
except ImportError:
    SafetyGate = None

try:
    from validation import AutoFixer
except ImportError:
    AutoFixer = None


@dataclass
class _RunRuntime:
    """Mutable data that belongs to one invocation only."""

    tool_call_count: int = 0
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = True


class AgentLoop:
    """Run an Agent until it returns text or the bounded fallback is reached."""

    def __init__(
        self,
        max_iterations: int = 10,
        short_term_memory: Optional[Any] = None,
        long_term_memory: Optional[Any] = None,
        max_tool_calls: int = 2,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.state_manager = StateManager()
        # Retained only for legacy direct callers that need history loading.
        # Writes are intentionally forbidden at this layer.
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.prompt_builder = prompt_builder or PromptBuilder()

        self.validator = ConstraintValidator() if ConstraintValidator else None
        self.auto_fixer = AutoFixer() if AutoFixer else None
        self.safety_gate = self._create_safety_gate()

    def _create_safety_gate(self) -> Any:
        if not SafetyGate:
            return None
        last_error: Optional[Exception] = None
        for args in ((self.validator, self.auto_fixer), (self.validator,), ()):
            try:
                return SafetyGate(*args)
            except TypeError:
                continue
            except Exception as exc:
                last_error = exc
                break
        error_type = type(last_error).__name__ if last_error else "TypeError"
        logger.error(f"Failed to initialize SafetyGate: error_type={error_type}")
        raise RuntimeError("medical_safety_gate_initialization_failed") from last_error

    async def run(
        self,
        agent: Any,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        """Execute one isolated run.

        ``session_id`` remains positional-compatible with the old API.
        ``request_context`` is the authoritative path used by the coordinator.
        """
        task_id = str(uuid.uuid4())
        state = self.state_manager.create_state(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=dict(input_data),
            max_iterations=self.max_iterations,
        )
        runtime = _RunRuntime()
        context = self._resolve_context(input_data, session_id, request_context)
        result: Dict[str, Any] = {}

        logger.info(f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}")
        try:
            state.status = TaskStatus.IN_PROGRESS

            blocked_response = await precheck_input(
                validator=self.validator,
                safety_gate=self.safety_gate,
                agent_id=agent.agent_id,
                question=context.raw_question,
                request_context=context,
            )
            if blocked_response:
                result = await self._complete_result(
                    agent=agent,
                    state=state,
                    context=context,
                    answer=blocked_response,
                    runtime=runtime,
                    warning="input_blocked_by_safety_gate",
                )
                return result

            messages = self._initialize_messages(
                agent,
                input_data,
                session_id=session_id,
                request_context=context,
            )
            tools_openai_format = agent.get_tools_for_llm()
            if self.validator and hasattr(self.validator, "filter_tool_definitions"):
                tools_openai_format = self.validator.filter_tool_definitions(
                    agent.agent_id,
                    tools_openai_format,
                )

            while state.should_continue():
                state.iteration += 1
                logger.debug(f"=== Iteration {state.iteration}/{state.max_iterations} ===")
                try:
                    llm_response: LLMResponse = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=tools_openai_format if runtime.tools_enabled else None,
                        tool_choice="auto" if runtime.tools_enabled else "none",
                        temperature=agent.config.get("temperature", 0.7),
                    )
                    state.add_intermediate_result({
                        "iteration": state.iteration,
                        "llm_response": {
                            "content": llm_response.content,
                            "tool_calls": [
                                {"name": call.name}
                                for call in llm_response.tool_calls
                            ],
                            "finish_reason": llm_response.finish_reason,
                        },
                    })

                    if llm_response.has_tool_calls() and runtime.tools_enabled:
                        remaining = max(0, self.max_tool_calls - runtime.tool_call_count)
                        if remaining == 0:
                            runtime.tools_enabled = False
                            messages.append({
                                "role": "user",
                                "content": "信息检索次数已达上限，请仅基于现有信息给出最终答复。",
                            })
                            continue

                        calls = list(llm_response.tool_calls[:remaining])
                        messages.append(self._create_assistant_message_with_tools(llm_response, calls))
                        for tool_call in calls:
                            validation = None
                            if self.validator and hasattr(self.validator, "validate_tool_call"):
                                validation = self.validator.validate_tool_call(
                                    agent.agent_id,
                                    tool_call.name,
                                )
                                if isinstance(validation, dict) and not validation.get("valid", True):
                                    logger.warning(f"Tool constraint warning: {validation.get('reason')}")

                            if isinstance(validation, dict) and validation.get("blocked"):
                                tool_result = {
                                    "error": "tool_call_blocked_by_policy",
                                    "reason": validation.get("reason", "not allowed"),
                                }
                                runtime.tool_call_count += 1
                                runtime.tool_trace.append({
                                    "tool_call_id": tool_call.id,
                                    "tool_name": tool_call.name,
                                    "success": False,
                                    "error_code": "tool_call_blocked_by_policy",
                                    "validation": {
                                        "valid": False,
                                        "blocked": True,
                                    },
                                    "blocked": True,
                                })
                                messages.append(agent.llm_client.create_tool_message(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    result=tool_result,
                                ))
                                continue

                            tool_result = await self._execute_scoped_tool(
                                agent=agent,
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments,
                                context=context,
                            )
                            runtime.tool_call_count += 1
                            success = not (
                                isinstance(tool_result, dict)
                                and (
                                    tool_result.get("success") is False
                                    or bool(tool_result.get("error"))
                                )
                            )
                            runtime.tool_trace.append({
                                "tool_call_id": tool_call.id,
                                "tool_name": tool_call.name,
                                "success": success,
                                "blocked": False,
                                "error_code": (
                                    str(tool_result.get("error", "tool_execution_failed"))[:80]
                                    if isinstance(tool_result, dict) and not success
                                    else None
                                ),
                                "validation": {
                                    "valid": bool(
                                        not isinstance(validation, dict)
                                        or validation.get("valid", True)
                                    ),
                                    "blocked": bool(
                                        isinstance(validation, dict)
                                        and validation.get("blocked", False)
                                    ),
                                },
                            })
                            messages.append(agent.llm_client.create_tool_message(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.name,
                                result=tool_result,
                            ))

                        if len(llm_response.tool_calls) > len(calls) or (
                            runtime.tool_call_count >= self.max_tool_calls
                        ):
                            runtime.tools_enabled = False
                            messages.append({
                                "role": "user",
                                "content": "信息检索次数已达上限，请仅基于已获得的结果给出最终答复。",
                            })
                        continue

                    answer = llm_response.content or ""
                    result = await self._complete_result(
                        agent=agent,
                        state=state,
                        context=context,
                        answer=answer,
                        runtime=runtime,
                    )
                    break

                except Exception as exc:
                    logger.error(
                        f"Agent iteration failed: iteration={state.iteration}, "
                        f"error_type={type(exc).__name__}"
                    )
                    if state.iteration >= state.max_iterations:
                        state.mark_failed(type(exc).__name__)
                        break

            if not state.is_completed() or state.status == TaskStatus.FAILED:
                result = await self._force_final_answer(
                    agent=agent,
                    state=state,
                    context=context,
                    messages=messages,
                    runtime=runtime,
                )

            logger.info(
                f"Agent Loop finished: status={state.status.value}, "
                f"iterations={state.iteration}, tools={runtime.tool_call_count}"
            )
            return result or state.final_result or {}
        except Exception as exc:
            logger.error(f"Agent Loop failed: error_type={type(exc).__name__}")
            state.mark_failed(type(exc).__name__)
            raise
        finally:
            # State contains request and tool payloads; release it immediately.
            self.state_manager.delete_state(task_id)

    def _resolve_context(
        self,
        input_data: Dict[str, Any],
        session_id: Optional[str],
        request_context: Optional[RequestContext],
    ) -> RequestContext:
        embedded = input_data.get("request_context")
        if request_context is None and isinstance(embedded, RequestContext):
            request_context = embedded
        if request_context is not None:
            return request_context

        question = input_data.get("question", input_data.get("query", str(input_data)))
        legacy_context = dict(input_data.get("context") or {})
        resolved_session = session_id or input_data.get("session_id")

        # Direct legacy callers may inject a memory instance into AgentLoop.
        # Load once into the immutable snapshot; never append at this layer.
        if self.short_term_memory and resolved_session and not any(
            key in legacy_context for key in ("recent_turns", "recent_history")
        ):
            try:
                history = self.short_term_memory.get_history(resolved_session, limit=5)
                if history:
                    legacy_context["recent_history"] = history
            except Exception as exc:
                logger.warning(
                    f"Could not load legacy conversation history: "
                    f"error_type={type(exc).__name__}"
                )

        return RequestContext.from_legacy(
            str(question),
            session_id=resolved_session,
            context=legacy_context,
            tenant_id=input_data.get("tenant_id"),
            user_id=input_data.get("user_id"),
            turn_id=input_data.get("turn_id"),
        )

    async def _execute_scoped_tool(
        self,
        *,
        agent: Any,
        tool_name: str,
        arguments: Dict[str, Any],
        context: RequestContext,
    ) -> Dict[str, Any]:
        """Execute memory Skills with server-bound scope, never LLM scope."""
        if tool_name == "search_history":
            if self.short_term_memory is None:
                return {
                    "success": False,
                    "error": "short_term_memory_unavailable",
                }
            try:
                limit = max(1, min(10, int(arguments.get("limit", 5))))
            except (TypeError, ValueError):
                limit = 5
            try:
                messages = await asyncio.to_thread(
                    self.short_term_memory.get_recent_messages,
                    context.session_id,
                    limit * 2 + 1,
                    tenant_id=context.tenant_id,
                    user_id=context.user_id,
                    turn_limit=limit,
                )
                canonical = [
                    {
                        "role": str(message.get("role", "")),
                        "content": redact_text(message.get("content", ""))[0],
                    }
                    for message in messages
                    if message.get("role") in {"system", "user", "assistant"}
                ]
                return {
                    "success": True,
                    "history": canonical,
                    "total_messages": len(canonical),
                }
            except Exception as exc:
                logger.error(
                    f"Scoped history search failed: error_type={type(exc).__name__}"
                )
                return {"success": False, "error": "history_search_failed"}

        if tool_name == "search_similar_cases":
            if not context.long_term_memory_consent:
                return {
                    "success": False,
                    "error": "long_term_memory_consent_required",
                }
            if self.long_term_memory is None or not getattr(
                self.long_term_memory, "enabled", False
            ):
                return {"success": False, "error": "long_term_memory_unavailable"}
            try:
                limit = max(1, min(5, int(arguments.get("max_results", 3))))
            except (TypeError, ValueError):
                limit = 3
            query = str(arguments.get("query") or context.raw_question)[:1000]
            search = self.long_term_memory.search_similar_sessions
            kwargs: Dict[str, Any] = {
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "min_score": 0.55,
                "current_session_id": context.session_id,
            }
            try:
                if "consent" in inspect.signature(search).parameters:
                    kwargs["consent"] = True
            except (TypeError, ValueError):
                pass
            try:
                results = await asyncio.to_thread(search, query, limit, **kwargs)
                safe_memories = memories_from_results(
                    results,
                    limit=limit,
                    min_score=0.55,
                )
                return {
                    "success": True,
                    "cases": [
                        {
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "score": memory.score,
                            "source": memory.source,
                            "metadata": thaw(memory.metadata),
                        }
                        for memory in safe_memories
                    ],
                    "total_found": len(safe_memories),
                }
            except Exception as exc:
                logger.error(
                    f"Scoped similar-case search failed: error_type={type(exc).__name__}"
                )
                return {"success": False, "error": "similar_case_search_failed"}

        return await agent.execute_tool(tool_name=tool_name, arguments=arguments)

    def _initialize_messages(
        self,
        agent: Any,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        request_context: Optional[RequestContext] = None,
    ) -> List[Dict[str, Any]]:
        """Compatibility wrapper around the sole prompt construction path."""
        context = request_context or self._resolve_context(input_data, session_id, None)
        return self.prompt_builder.build(
            system_prompt=agent.get_system_prompt(),
            context=context,
        )

    async def _complete_result(
        self,
        *,
        agent: Any,
        state: Any,
        context: RequestContext,
        answer: str,
        runtime: _RunRuntime,
        warning: Optional[str] = None,
    ) -> Dict[str, Any]:
        final_answer = answer or "抱歉，未能生成有效答复。"

        if self.validator and hasattr(self.validator, "validate_output"):
            validation = self.validator.validate_output(agent.agent_id, final_answer)
            if isinstance(validation, dict) and not validation.get("valid", True):
                logger.warning(f"Output constraint violations: {validation.get('violations')}")
                if self.auto_fixer and validation.get("auto_fixable"):
                    final_answer = self.auto_fixer.fix_output(
                        final_answer,
                        validation.get("auto_fixable", []),
                    )

        final_answer = await finalize_output(
            validator=self.validator,
            safety_gate=self.safety_gate,
            agent_id=agent.agent_id,
            question=context.raw_question,
            output=final_answer,
            request_context=context,
        )

        result: Dict[str, Any] = {
            "answer": final_answer,
            "iterations": state.iteration,
            "agent_id": agent.agent_id,
            "tool_call_count": runtime.tool_call_count,
            "tool_trace": list(runtime.tool_trace),
            "session_id": context.session_id,
            "turn_id": context.turn_id,
        }
        if warning:
            result["warning"] = warning
        if hasattr(agent, "post_process_result"):
            result = await agent.post_process_result(result, final_answer)
        state.mark_completed(result)
        return result

    async def _force_final_answer(
        self,
        *,
        agent: Any,
        state: Any,
        context: RequestContext,
        messages: List[Dict[str, Any]],
        runtime: _RunRuntime,
    ) -> Dict[str, Any]:
        try:
            messages.append({
                "role": "user",
                "content": "请停止调用工具，基于现有信息提供最终答复。",
            })
            response = await agent.llm_client.chat_with_tools(
                messages=messages,
                tools=None,
                temperature=agent.config.get("temperature", 0.7),
            )
            return await self._complete_result(
                agent=agent,
                state=state,
                context=context,
                answer=response.content or "抱歉，未能完成任务。",
                runtime=runtime,
                warning="max_iterations_reached",
            )
        except Exception as exc:
            logger.error(
                f"Failed to generate fallback answer: error_type={type(exc).__name__}"
            )
            return await self._complete_result(
                agent=agent,
                state=state,
                context=context,
                answer="抱歉，系统在处理您的问题时遇到了问题。建议简化问题或稍后重试。",
                runtime=runtime,
                warning="fallback_after_error",
            )

    def _create_assistant_message_with_tools(
        self,
        llm_response: LLMResponse,
        tool_calls: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        calls = tool_calls if tool_calls is not None else list(llm_response.tool_calls)
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": llm_response.content or None,
        }
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in calls
            ]
        return message


__all__ = ["AgentLoop"]

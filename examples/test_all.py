#!/usr/bin/env python3
"""Deterministic offline tests for MediX Agent Swarm.

This suite uses only Python's standard library.  It does not initialize a live
LLM, Redis, Mem0, Milvus, an embedding model, or a network client.  Optional
logging is stubbed only when ``loguru`` is unavailable.  A failure returns a
non-zero process status; skipped tests are treated as a suite failure.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import ModuleType, SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_optional_logging_stub() -> None:
    try:
        import loguru  # noqa: F401
        return
    except ImportError:
        pass

    class NullLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    module = ModuleType("loguru")
    module.logger = NullLogger()
    sys.modules["loguru"] = module


_install_optional_logging_stub()


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class FakeCompletions:
    def __init__(self) -> None:
        self.requests = []
        self.response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="offline response", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class FakeOpenAICompatibleClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FakeMemoryClient:
    """In-memory Mem0-shaped fake; no sockets or external writes."""

    def __init__(self) -> None:
        self.add_calls = []
        self.search_calls = []
        self.search_results = []
        self.records = {}
        self.deleted = []

    def add(self, **kwargs):
        self.add_calls.append(kwargs)
        memory_id = f"memory-{len(self.add_calls)}"
        self.records[memory_id] = {
            "id": memory_id,
            "metadata": dict(kwargs.get("metadata") or {}),
        }
        return {"id": memory_id}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return list(self.search_results)

    def get(self, memory_id=None, **_kwargs):
        return self.records.get(memory_id, {})

    def delete(self, memory_id=None, **_kwargs):
        self.deleted.append(memory_id)
        return True


class TestConfigurationAndProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = importlib.import_module("core.llm_client")

    def test_module_import_does_not_require_openai_package(self):
        self.assertTrue(hasattr(self.module, "LLMClient"))

    def test_explicit_configuration_wins_and_zero_temperature_is_preserved(self):
        config = self.module.load_llm_config(
            {
                "api_key": "offline-placeholder",
                "model_name": "explicit-model",
                "base_url": "https://example.invalid/v1",
                "temperature": 0,
                "max_tokens": 64,
            },
            environ={
                "MEDIX_LLM_MODEL": "environment-model",
                "MEDIX_LLM_TEMPERATURE": "1.0",
            },
        )
        self.assertEqual(config["model_name"], "explicit-model")
        self.assertEqual(config["temperature"], 0.0)

    def test_live_client_is_replaceable_with_offline_fake(self):
        fake = FakeOpenAICompatibleClient()
        client = self.module.LLMClient(
            config={
                "model_name": "fake-model",
                "base_url": "https://example.invalid/v1",
                "temperature": 0.7,
                "max_tokens": 99,
            },
            client=fake,
        )
        result = asyncio.run(
            client.chat([{"role": "user", "content": "test"}], temperature=0)
        )
        self.assertEqual(result, "offline response")
        self.assertEqual(fake.completions.requests[0]["temperature"], 0)

    def test_tool_calls_are_normalized_without_network(self):
        fake = FakeOpenAICompatibleClient()
        fake.completions.response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="assess_risk",
                                    arguments='{"symptom": "头痛"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )
        client = self.module.LLMClient(
            config={"model_name": "fake", "base_url": "https://example.invalid/v1"},
            client=fake,
        )
        response = asyncio.run(client.chat_with_tools([], tools=[{"type": "function"}]))
        self.assertTrue(response.has_tool_calls())
        self.assertEqual(response.tool_calls[0].arguments, {"symptom": "头痛"})

    def test_unsafe_remote_http_endpoint_is_rejected(self):
        with self.assertRaises(self.module.LLMConfigurationError):
            self.module.LLMClient(
                config={"model_name": "fake", "base_url": "http://example.com/v1"},
                client=FakeOpenAICompatibleClient(),
            )

    def test_config_file_cannot_escape_project(self):
        with self.assertRaises(self.module.LLMConfigurationError):
            self.module.load_llm_config(config_file="../config.py", environ={})


class TestSafetyConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        constraints = importlib.import_module("constraints")
        cls.validator = constraints.ConstraintValidator()
        cls.gate = constraints.SafetyGate(cls.validator)

    def test_constraint_files_are_standard_library_parseable(self):
        for relative, root_key in (
            ("constraints/agent_constraints.yaml", "agents"),
            ("constraints/swarm_constraints.yaml", "swarm"),
        ):
            policy = json.loads(_read(relative))
            self.assertIn(root_key, policy)

    def test_emergency_input_is_short_circuited(self):
        result = self.gate.precheck_input("我突然胸痛、呼吸困难")
        self.assertTrue(result["blocked"])
        self.assertFalse(result["safe_to_continue"])
        self.assertIn("120", result["response"])
        self.assertIn("免责声明", result["response"])

    def test_negated_emergency_symptom_does_not_trigger(self):
        result = self.gate.precheck_input("目前没有胸痛，也无呼吸困难")
        self.assertFalse(result["blocked"])

    def test_personal_identifiers_are_detected_without_echoing_values(self):
        result = self.gate.precheck_input("手机号是13800138000，邮箱a@example.com")
        self.assertEqual(set(result["privacy_risks"]), {"mainland_phone", "email"})
        self.assertNotIn("13800138000", result["privacy_warning"])

    def test_disallowed_tool_call_is_blocking(self):
        result = self.validator.validate_tool_call("consultation_agent", "disease_code")
        self.assertFalse(result["valid"])
        self.assertTrue(result["blocked"])

    def test_unknown_agent_is_fail_closed(self):
        result = self.validator.validate_tool_call("invented_agent", "assess_risk")
        self.assertTrue(result["blocked"])

    def test_safe_tool_call_is_allowed(self):
        result = self.validator.validate_tool_call("ConsultationAgent", "assess_risk")
        self.assertTrue(result["valid"])
        self.assertFalse(result["blocked"])

    def test_tool_schemas_can_be_filtered_before_model_call(self):
        schemas = [
            {"type": "function", "function": {"name": "assess_risk"}},
            {"type": "function", "function": {"name": "disease_code"}},
        ]
        filtered = self.validator.filter_tool_definitions("consultation_agent", schemas)
        self.assertEqual(
            [item["function"]["name"] for item in filtered],
            ["assess_risk"],
        )

    def test_final_gate_repairs_diagnosis_prescription_privacy_and_disclaimer(self):
        raw = "您患有高血压，建议服用某药20mg。手机号13800138000。"
        result = self.gate.finalize_output(raw, user_input="我的血压偏高")
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["released"])
        self.assertNotIn("您患有", result["output"])
        self.assertNotIn("13800138000", result["output"])
        self.assertIn("免责声明", result["output"])

    def test_specific_drug_instruction_without_dose_is_also_removed(self):
        result = self.gate.finalize_output("建议服用阿司匹林。", user_input="普通咨询")
        self.assertTrue(result["gate_passed"])
        self.assertNotIn("建议服用阿司匹林", result["output"])
        self.assertIn("医生", result["output"])

    def test_emergency_final_gate_never_releases_generated_diagnosis(self):
        raw = "您患有心梗，建议在家观察。"
        result = self.gate.finalize_output(raw, user_input="我胸痛且喘不上气")
        self.assertTrue(result["blocked"])
        self.assertFalse(result["released"])
        self.assertIn("120", result["output"])
        self.assertNotIn("心梗", result["output"])

    def test_missing_research_citation_cannot_be_auto_fabricated(self):
        raw = "研究证明该方法有效。以上仅供参考，不能替代专业医生。"
        result = self.gate.finalize_output(raw, agent_id="research_agent")
        self.assertTrue(result["blocked"])
        self.assertFalse(result["released"])
        self.assertIn("未通过医疗安全校验", result["output"])

    def test_single_and_swarm_use_the_same_final_policy(self):
        answer = "建议记录症状变化。以上信息仅供参考，不能替代专业医生。"
        single = self.gate.finalize_output(answer, agent_id="consultation_agent")
        swarm = self.gate.finalize_output(answer, agent_id="final_answer")
        self.assertTrue(single["gate_passed"])
        self.assertTrue(swarm["gate_passed"])
        self.assertEqual(single["output"], swarm["output"])

    def test_high_risk_plan_requires_diagnostic_agent(self):
        result = self.validator.validate_task_decomposition(
            "我胸痛",
            [{"id": "one", "description": "提供建议", "assigned_agent": "consultation_agent"}],
        )
        self.assertTrue(result["blocked"])
        self.assertIn("diagnostic_agent", result["missing_required_agents"])

    def test_unimplemented_swarm_modes_are_explicitly_blocked(self):
        self.assertTrue(self.validator.validate_collaboration_mode("parallel")["valid"])
        self.assertTrue(self.validator.validate_collaboration_mode("sequential")["blocked"])
        self.assertTrue(self.validator.validate_collaboration_mode("invented")["blocked"])

    def test_swarm_evidence_and_conflict_policies_are_enforced(self):
        contribution = {"result": {"answer": "这是一个长度足够但没有任何可核验来源的分析结果。" * 3}}
        answer = "建议由医生结合检查评估。以上信息仅供参考，不能替代专业医生。"
        result = self.validator.validate_swarm_result(
            [contribution],
            answer,
            disagreements=[{"left": "A", "right": "B"}],
        )
        self.assertTrue(result["blocked"])
        self.assertTrue(any("证据" in issue for issue in result["issues"]))
        self.assertTrue(any("冲突" in issue for issue in result["issues"]))

        evidenced = {
            "result": {
                "answer": "有来源支持的分析结果。" * 6,
                "evidence": [{"source": "verified-local-guideline", "year": 2024}],
            }
        }
        approved = self.validator.validate_swarm_result([evidenced], answer)
        self.assertFalse(approved["blocked"])


class TestContextAndPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context_module = importlib.import_module("core.conversation_context")
        cls.prompt_module = importlib.import_module("core.prompt_builder")

    def test_only_complete_turns_enter_context(self):
        messages = [
            {"role": "assistant", "content": "orphan"},
            {"role": "user", "content": "first"},
            {"role": "tool", "content": "internal trace"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "unfinished"},
        ]
        turns = self.context_module.complete_turns_from_messages(messages, limit=5)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].user, "first")
        self.assertEqual(turns[0].assistant, "answer one")

    def _context(self):
        return self.context_module.RequestContext(
            tenant_id="tenant-a",
            user_id="user-a",
            session_id="session-a",
            turn_id="turn-2",
            raw_question="那我接下来怎么办？",
            patient_profile={"age": 35},
            rolling_summary="用户此前询问头痛。",
            recent_turns=(
                self.context_module.ConversationTurn(
                    user="我头痛两天了",
                    assistant="请记录伴随症状",
                    turn_id="turn-1",
                ),
            ),
            retrieved_memories=(
                self.context_module.RetrievedMemory(
                    content="用户曾报告偏头痛史",
                    score=0.9,
                    memory_id="memory-1",
                ),
            ),
        )

    def test_prompt_injects_each_history_item_once(self):
        context = self._context()
        messages = self.prompt_module.PromptBuilder().build(
            system_prompt="system",
            context=context,
        )
        rendered = "\n".join(str(message.get("content", "")) for message in messages)
        self.assertEqual(rendered.count("我头痛两天了"), 1)
        self.assertEqual(rendered.count("请记录伴随症状"), 1)
        self.assertEqual(rendered.count("那我接下来怎么办？"), 1)
        self.assertEqual(messages[-1]["role"], "user")

    def test_worker_context_is_an_immutable_scoped_view(self):
        original = self._context()
        worker = original.for_task(
            "评估风险",
            collaboration_results=({"agent_id": "consultation_agent", "answer": "result"},),
        )
        self.assertEqual(worker.session_id, original.session_id)
        self.assertEqual(worker.raw_question, original.raw_question)
        self.assertEqual(worker.task_instruction, "评估风险")
        self.assertEqual(original.task_instruction, "")
        with self.assertRaises(Exception):
            worker.patient_profile["age"] = 99

    def test_single_swarm_single_context_continuity(self):
        first = self._context()
        worker = first.for_task("分析本轮问题")
        synthesis = first.for_task(
            "汇总回答", collaboration_results=({"agent_id": "worker", "answer": "done"},)
        )
        next_turn = self.context_module.RequestContext(
            tenant_id=first.tenant_id,
            user_id=first.user_id,
            session_id=first.session_id,
            turn_id="turn-3",
            raw_question="刚才的建议还适用吗？",
            recent_turns=first.recent_turns
            + (
                self.context_module.ConversationTurn(
                    user=first.raw_question,
                    assistant="这是汇总后的答复",
                    turn_id=first.turn_id,
                ),
            ),
        )
        self.assertEqual(worker.storage_session_id, synthesis.storage_session_id)
        prompt = self.prompt_module.PromptBuilder().build(
            system_prompt="system", context=next_turn
        )
        rendered = "\n".join(message["content"] for message in prompt)
        self.assertIn(first.raw_question, rendered)
        self.assertIn("这是汇总后的答复", rendered)
        self.assertIn(next_turn.raw_question, rendered)

    def test_context_is_explicitly_propagated_across_public_boundaries(self):
        expected = {
            "core/agent_loop.py": {"run": "request_context"},
            "agents/base_agent.py": {
                "process": "request_context",
                "process_subtask": "request_context",
            },
            "swarm/lead_agent.py": {
                "assess_and_decompose": "request_context",
                "synthesize_results": "request_context",
            },
        }
        for relative, functions in expected.items():
            tree = ast.parse(_read(relative), filename=relative)
            definitions = {
                node.name: [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for function_name, argument in functions.items():
                self.assertIn(function_name, definitions, f"missing {relative}:{function_name}")
                self.assertIn(argument, definitions[function_name])

        coordinator_tree = ast.parse(_read("swarm/swarm_coordinator.py"))
        coordinator_process = next(
            node
            for node in ast.walk(coordinator_tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "process"
        )
        arguments = [
            arg.arg for arg in (*coordinator_process.args.args, *coordinator_process.args.kwonlyargs)
        ]
        for name in ("session_id", "tenant_id", "user_id", "turn_id"):
            self.assertIn(name, arguments)

    def test_agent_loop_never_persists_formatted_prompts(self):
        source = _read("core/agent_loop.py")
        self.assertNotIn(".add_message(", source)
        self.assertNotIn(".add_turn(", source)

    def test_safety_hook_failure_is_fail_closed(self):
        hooks = importlib.import_module("core.safety_hooks")

        class ExplodingValidator:
            def precheck_input(self, text):
                raise RuntimeError("offline test")

            def gate_output(self, agent_id, output, user_input=None):
                raise RuntimeError("offline test")

        prechecked = asyncio.run(
            hooks.precheck_input(
                validator=ExplodingValidator(),
                safety_gate=None,
                agent_id="final_answer",
                question="ordinary question",
            )
        )
        finalized = asyncio.run(
            hooks.finalize_output(
                validator=ExplodingValidator(),
                safety_gate=None,
                agent_id="final_answer",
                question="ordinary question",
                output="unsafe unchecked output",
            )
        )
        self.assertEqual(prechecked, hooks.DEFAULT_BLOCK_RESPONSE)
        self.assertEqual(finalized, hooks.DEFAULT_BLOCK_RESPONSE)

    def test_coordinator_binds_external_and_local_persistence_to_consent(self):
        source = _read("swarm/swarm_coordinator.py")
        self.assertIn('if "consent" in inspect.signature(search).parameters', source)
        self.assertIn('search_kwargs["consent"] = True', source)
        self.assertIn('if "consent" in inspect.signature(save).parameters', source)
        self.assertIn('save_kwargs["consent"] = True', source)
        self.assertIn("request_context.long_term_memory_consent", source)


class TestMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory = importlib.import_module("memory")

    def test_long_term_dedup_uses_content_not_missing_fields(self):
        manager = self.memory.MemoryEntropyManager()
        sessions = [
            {"memory_id": "1", "content": "first distinct memory", "score": 0.9},
            {"memory_id": "2", "content": "second distinct memory", "score": 0.8},
            {"memory_id": "3", "content": "third distinct memory", "score": 0.7},
            {"memory_id": "4", "content": "first distinct memory", "score": 0.6},
        ]
        result = manager.deduplicate_sessions(sessions)
        self.assertEqual([item["memory_id"] for item in result], ["1", "2", "3"])

    def test_short_term_memory_is_isolated_by_tenant_user_and_session(self):
        memory = self.memory.ShortTermMemory(storage_type="memory")
        memory.add_turn("same", "tenant A question", "tenant A answer", tenant_id="a", user_id="u")
        memory.add_turn("same", "tenant B question", "tenant B answer", tenant_id="b", user_id="u")
        first = memory.get_history("same", tenant_id="a", user_id="u")
        second = memory.get_history("same", tenant_id="b", user_id="u")
        self.assertIn("tenant A question", str(first))
        self.assertNotIn("tenant B question", str(first))
        self.assertIn("tenant B question", str(second))

    def test_tool_traces_do_not_pollute_conversation_transcript(self):
        memory = self.memory.ShortTermMemory(storage_type="memory")
        memory.add_turn("s", "question", "answer", tenant_id="t", user_id="u")
        memory.add_message(
            "s", "tool", "private tool payload", tenant_id="t", user_id="u", message_type="tool"
        )
        history = memory.get_history("s", tenant_id="t", user_id="u")
        traces = memory.get_tool_traces("s", tenant_id="t", user_id="u")
        self.assertNotIn("private tool payload", str(history))
        self.assertIn("private tool payload", str(traces))

    def test_short_term_concurrent_turn_appends_are_complete(self):
        memory = self.memory.ShortTermMemory(
            storage_type="memory", max_stored_messages=200, max_context_chars=50000
        )

        def add(index):
            memory.add_turn(
                "concurrent",
                f"question-{index}",
                f"answer-{index}",
                tenant_id="tenant",
                user_id="user",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(add, range(40)))
        history = memory.get_history(
            "concurrent", limit=50, tenant_id="tenant", user_id="user", char_budget=50000
        )
        roles = [message["role"] for message in history if message["role"] != "system"]
        self.assertEqual(len(roles), 80)
        self.assertEqual(roles, [role for _ in range(40) for role in ("user", "assistant")])

    def test_external_phi_requires_configuration_and_per_write_consent(self):
        client = FakeMemoryClient()
        memory = self.memory.LongTermMemory(client=client, allow_external_phi=True)
        rejected = memory.add_session_summary(
            "s", "question", "answer", tenant_id="t", user_id="u", consent=False
        )
        accepted = memory.add_session_summary(
            "s", "question", "answer", tenant_id="t", user_id="u", consent=True
        )
        self.assertIsNone(rejected)
        self.assertIsNotNone(accepted)
        self.assertEqual(len(client.add_calls), 1)

    def test_external_search_also_requires_per_request_consent(self):
        client = FakeMemoryClient()
        memory = self.memory.LongTermMemory(client=client, allow_external_phi=True)
        result = memory.search_similar_sessions("query", tenant_id="t", user_id="u")
        self.assertEqual(result, [])
        self.assertEqual(client.search_calls, [])

    def test_long_term_backend_user_ids_are_tenant_scoped_and_hashed(self):
        client = FakeMemoryClient()
        memory = self.memory.LongTermMemory(client=client, allow_external_phi=True)
        memory.add_session_summary(
            "same", "q", "a", tenant_id="tenant-a", user_id="same-user", consent=True
        )
        memory.add_session_summary(
            "same", "q", "a", tenant_id="tenant-b", user_id="same-user", consent=True
        )
        first, second = [call["user_id"] for call in client.add_calls]
        self.assertNotEqual(first, second)
        self.assertNotIn("same-user", first)
        self.assertNotIn("tenant-a", first)

    def test_long_term_search_filters_cross_tenant_low_score_and_expired_data(self):
        client = FakeMemoryClient()
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        client.search_results = [
            {
                "id": "ok",
                "memory": "relevant memory",
                "score": 0.9,
                "metadata": {
                    "tenant_id": "t1", "user_id": "u1", "session_id": "old", "expires_at": future
                },
            },
            {
                "id": "leak",
                "memory": "other tenant",
                "score": 0.99,
                "metadata": {"tenant_id": "t2", "user_id": "u1", "expires_at": future},
            },
            {
                "id": "low",
                "memory": "low score",
                "score": 0.1,
                "metadata": {"tenant_id": "t1", "user_id": "u1", "expires_at": future},
            },
            {
                "id": "expired",
                "memory": "expired memory",
                "score": 0.95,
                "metadata": {"tenant_id": "t1", "user_id": "u1", "expires_at": past},
            },
        ]
        memory = self.memory.LongTermMemory(
            client=client,
            min_similarity_score=0.55,
            allow_external_phi=True,
        )
        result = memory.search_similar_sessions(
            "query", tenant_id="t1", user_id="u1", consent=True
        )
        self.assertEqual([item["memory_id"] for item in result], ["ok"])

    def _summary(self, session_id="session", turn_id="turn", summary_id="summary"):
        return self.memory.SessionSummary(
            session_id=session_id,
            question="question",
            context={"age": 30},
            timestamp=datetime.now(timezone.utc),
            agents_participated=[],
            subtasks_created=0,
            subtasks_completed=0,
            events_count=0,
            final_answer="complete answer",
            key_findings=[],
            lessons_learned=[],
            performance=self.memory.PerformanceMetrics(0, 0, 0, 0, 0),
            tenant_id="tenant",
            user_id="user",
            turn_id=turn_id,
            summary_id=summary_id,
        )

    def test_session_summary_roundtrip_is_lossless_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.memory.SessionSummaryManager(directory)
            first = self._summary(turn_id="one", summary_id="one")
            second = self._summary(turn_id="two", summary_id="two")
            with self.assertRaises(PermissionError):
                manager.save_summary(first)
            self.assertEqual(list(Path(directory).rglob("*.json")), [])
            first_path = manager.save_summary(first, consent=True)
            second_path = manager.save_summary(second, consent=True)
            loaded = manager.load_summary(
                first.session_id,
                tenant_id=first.tenant_id,
                user_id=first.user_id,
                summary_id=first.summary_id,
            )
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(loaded.to_dict(), first.to_dict())
            self.assertEqual(len(manager.list_summaries("session", tenant_id="tenant", user_id="user")), 2)

    def test_session_summary_path_cannot_escape_base_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = self.memory.SessionSummaryManager(directory)
            summary = self._summary(
                session_id="../../escape", turn_id="../../../turn", summary_id="../../id"
            )
            path = manager.save_summary(summary, consent=True).resolve()
            self.assertIn(Path(directory).resolve(), path.parents)
            self.assertFalse((Path(directory).parent / "escape").exists())


class TestRepositoryContracts(unittest.TestCase):
    def test_all_python_sources_parse(self):
        errors = []
        for path in ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                errors.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_regular_modules_exist_and_timestamp_backups_are_absent(self):
        self.assertTrue((ROOT / "swarm/events.py").is_file())
        self.assertTrue((ROOT / "validation/auto_fixer.py").is_file())
        self.assertFalse((ROOT / "swarm/events_20260428_231035.py").exists())
        self.assertFalse((ROOT / "validation/auto_fixer_20260428_231043.py").exists())

    def test_setup_declares_entrypoint_and_constraint_package_data(self):
        source = _read("setup.py")
        tree = ast.parse(source, filename="setup.py")
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setup"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertIn("entry_points", keywords)
        self.assertIn("package_data", keywords)
        self.assertIn("data_files", keywords)

    def test_local_settings_are_secret_free_and_not_permission_broadening(self):
        settings = json.loads(_read(".claude/settings.local.json"))
        self.assertEqual(settings.get("permissions", {}).get("allow"), [])
        ignored = _read(".gitignore")
        self.assertIn("/.claude/settings.local.json", ignored)
        self.assertIn("memory/swarm/session_summaries/", ignored)

    def test_repository_contains_no_known_secret_shapes_or_machine_paths(self):
        forbidden = {
            "old machine path": re.compile("/Users/" + "saint" + "geo"),
            "Mem0 key": re.compile(r"m0-[A-Za-z0-9]{20,}"),
            "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        }
        findings = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".md", ".json", ".yaml", ".txt"} and path.name not in {
                ".gitignore", ".env.example"
            }:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for label, pattern in forbidden.items():
                if pattern.search(text):
                    findings.append(f"{path.relative_to(ROOT)}: {label}")
        self.assertEqual(findings, [], "\n".join(findings))

    def test_readme_has_no_unverified_success_claim_and_documents_offline_default(self):
        readme = _read("README.md")
        self.assertNotIn("100%", readme)
        self.assertIn("MEDIX_ALLOW_MODEL_DOWNLOAD", readme)
        self.assertIn("local_files_only", readme)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(f"ERROR: {len(result.skipped)} skipped tests are not accepted as passes", file=sys.stderr)
        return 1
    if result.testsRun == 0:
        print("ERROR: no tests were discovered", file=sys.stderr)
        return 1
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

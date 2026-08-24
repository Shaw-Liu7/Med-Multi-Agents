"""Fail-closed runtime constraints for the medical assistant.

The ``*.yaml`` files in this package intentionally use JSON syntax (JSON is a
valid subset of YAML), which keeps the safety layer usable with Python's
standard library alone.  A malformed or missing policy is a startup error: a
medical safety layer must never silently disable itself.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


class ConstraintDefinitionError(RuntimeError):
    """Raised when a constraint definition cannot be loaded or validated."""


def _load_policy(path: Path, expected_root: str) -> Dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConstraintDefinitionError(f"Invalid constraint policy: {path.name}") from exc
    if not isinstance(policy, dict) or expected_root not in policy:
        raise ConstraintDefinitionError(
            f"Constraint policy {path.name} must contain '{expected_root}'"
        )
    return policy


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


class ConstraintValidator:
    """Validate input triage, tool permissions, outputs, and Swarm plans."""

    _DIAGNOSIS_PATTERNS = [
        re.compile(pattern)
        for pattern in (
            r"(?:您|你)(?:患有|得了|就是)[^，。；\n]{1,40}",
            r"(?:已经|可以)?确诊为[^，。；\n]{1,40}",
            r"(?:明确诊断|最终诊断)(?:为|是)[^，。；\n]{1,40}",
            r"这(?:就是|是)[^，。；\n]{1,25}(?:病|炎|癌|综合征)",
            r"(?:肯定|一定|百分之百)是[^，。；\n]{1,40}",
        )
    ]
    _PRESCRIPTION_PATTERNS = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in (
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?(?:服用|使用|口服|注射)[^，。；\n]{0,30}\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|毫升)",
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?(?:服用|口服|注射)[^，。；\n]{1,35}",
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?使用[^，。；\n]{0,20}(?:阿司匹林|布洛芬|对乙酰氨基酚|抗生素|激素|胰岛素)",
            r"(?:每日|每天|一日)\s*\d+\s*次[^，。；\n]{0,30}(?:每次)?\s*\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|毫升)",
            r"(?:自行|自己)(?:服药|用药|治疗|停药|加量|减量)",
        )
    ]
    _TREATMENT_RECOMMENDATION = re.compile(
        r"(?:建议|推荐|应当)(?:您|你)?(?:"
        r"(?:接受|进行|采用|开始)[^，。；\n]{0,35}(?:治疗|手术|疗法)"
        r"|(?:服用|使用|停用)[^，。；\n]{0,35}(?:药|片|针|mg|毫克)"
        r")"
    )
    _GUARANTEE_PATTERNS = [
        "保证治愈",
        "一定能治好",
        "百分之百治愈",
        "绝对有效",
        "不会复发",
        "完全没有风险",
    ]
    _DISRESPECTFUL_TERMS = ["愚蠢", "活该", "矫情", "别废话", "蠢", "无知"]
    _FEAR_TERMS = ["没救了", "等死", "必死无疑", "非常可怕", "灾难性后果"]
    _JARGON_TERMS = ["鉴别诊断", "预后", "病理生理", "依从性", "禁忌证", "合并症"]
    _EVIDENCE_MARKERS = [
        re.compile(r"https?://", re.IGNORECASE),
        re.compile(r"\bdoi\s*:\s*10\.\d{4,9}/", re.IGNORECASE),
        re.compile(r"(?:指南|共识|系统综述|随机对照试验).{0,30}(?:19|20)\d{2}"),
        re.compile(r"\[(?:\d+|[A-Za-z]+\s+et\s+al\.)\]"),
    ]

    def __init__(
        self,
        agent_constraints_file: Optional[str | Path] = None,
        swarm_constraints_file: Optional[str | Path] = None,
    ):
        package_dir = Path(__file__).resolve().parent
        agent_path = (
            Path(agent_constraints_file).expanduser().resolve()
            if agent_constraints_file
            else package_dir / "agent_constraints.yaml"
        )
        swarm_path = (
            Path(swarm_constraints_file).expanduser().resolve()
            if swarm_constraints_file
            else package_dir / "swarm_constraints.yaml"
        )

        self.agent_constraints = _load_policy(agent_path, "agents")
        self.swarm_constraints = _load_policy(swarm_path, "swarm")
        self._validate_policy_schema()

        self._agent_aliases: Dict[str, str] = {}
        for canonical, policy in self.agent_constraints["agents"].items():
            self._agent_aliases[canonical.casefold()] = canonical
            for alias in policy.get("aliases", []):
                self._agent_aliases[str(alias).casefold()] = canonical

        privacy = self.agent_constraints.get("privacy", {}).get("identifier_patterns", {})
        try:
            self._identifier_patterns = {
                name: re.compile(pattern) for name, pattern in privacy.items()
            }
        except re.error as exc:
            raise ConstraintDefinitionError("Invalid privacy regular expression") from exc

    def _validate_policy_schema(self) -> None:
        agents = self.agent_constraints.get("agents")
        if not isinstance(agents, dict) or not agents:
            raise ConstraintDefinitionError("At least one agent policy is required")
        for agent_id, policy in agents.items():
            if not isinstance(policy, dict):
                raise ConstraintDefinitionError(f"Invalid policy for {agent_id}")
            for key in ("allowed_tools", "forbidden_actions", "output_constraints"):
                if key not in policy or not isinstance(policy[key], list):
                    raise ConstraintDefinitionError(f"{agent_id}.{key} must be a list")

        swarm = self.swarm_constraints.get("swarm", {})
        for numeric_key in ("max_agents_per_task", "max_parallel_tasks", "timeout_seconds"):
            value = swarm.get(numeric_key)
            if not isinstance(value, int) or value <= 0:
                raise ConstraintDefinitionError(f"swarm.{numeric_key} must be positive")

    def normalize_agent_id(self, agent_id: str) -> Optional[str]:
        """Resolve class names and legacy aliases to one policy identifier."""

        return self._agent_aliases.get(str(agent_id or "").casefold())

    def _agent_policy(self, agent_id: str) -> Optional[Dict[str, Any]]:
        canonical = self.normalize_agent_id(agent_id)
        if canonical is None:
            return None
        return self.agent_constraints["agents"][canonical]

    def _is_negated(self, text: str, start: int) -> bool:
        triage = self.agent_constraints.get("emergency_triage", {})
        prefix = text[max(0, start - 8) : start]
        return any(term in prefix for term in triage.get("negation_terms", []))

    def _emergency_matches(self, text: str) -> List[str]:
        matches: List[str] = []
        triage = self.agent_constraints.get("emergency_triage", {})
        for keyword in triage.get("emergency_keywords", []):
            offset = 0
            while True:
                index = text.find(keyword, offset)
                if index < 0:
                    break
                if not self._is_negated(text, index):
                    matches.append(keyword)
                    break
                offset = index + len(keyword)
        return _unique(matches)

    def _privacy_matches(self, text: str) -> List[str]:
        return [name for name, pattern in self._identifier_patterns.items() if pattern.search(text)]

    def precheck_input(self, text: str) -> Dict[str, Any]:
        """Run deterministic emergency triage before any model or tool call.

        Emergency input is deliberately short-circuited to a static response;
        no generated diagnosis is needed before urgent care is recommended.
        Privacy findings are warnings rather than blockers, but callers should
        avoid persisting the original text until it has been redacted.
        """

        normalized = str(text or "").strip()
        emergency_terms = self._emergency_matches(normalized)
        privacy_risks = self._privacy_matches(normalized)
        urgent = bool(emergency_terms)
        triage = self.agent_constraints.get("emergency_triage", {})

        return {
            "valid": not urgent,
            "safe_to_continue": not urgent,
            "blocked": urgent,
            "risk_level": "emergency" if urgent else "normal",
            "matched_symptoms": emergency_terms,
            "privacy_risks": privacy_risks,
            "privacy_warning": (
                "输入可能包含手机号、身份证号或邮箱；请在日志和记忆写入前脱敏。"
                if privacy_risks
                else ""
            ),
            "response": triage.get("emergency_response", "") if urgent else "",
        }

    # Compatibility alias used by integrations that name this operation triage.
    validate_input = precheck_input

    def validate_tool_call(self, agent_id: str, tool_name: str) -> Dict[str, Any]:
        """Enforce the per-agent tool allowlist.

        ``valid=False`` always means the caller must *not* execute the tool.
        The old warning-only behavior was unsafe and is intentionally removed.
        """

        canonical = self.normalize_agent_id(agent_id)
        policy = self._agent_policy(agent_id)
        if policy is None:
            reason = f"未知 Agent {agent_id!r} 没有工具权限"
            return {
                "valid": False,
                "allowed": False,
                "blocked": True,
                "severity": "error",
                "reason": reason,
            }

        allowed_tools = policy.get("allowed_tools", [])
        deny_all = bool(policy.get("deny_all_tools", False))
        if deny_all or tool_name not in allowed_tools:
            reason = f"工具 {tool_name!r} 不在 {canonical} 的允许列表中"
            logger.warning("Blocked disallowed tool call: agent=%s tool=%s", canonical, tool_name)
            return {
                "valid": False,
                "allowed": False,
                "blocked": True,
                "severity": "error",
                "reason": reason,
            }

        return {
            "valid": True,
            "allowed": True,
            "blocked": False,
            "severity": "none",
            "reason": "",
        }

    def get_allowed_tools(self, agent_id: str) -> List[str]:
        """Return a copy of the role allowlist; unknown/deny-all agents get none."""

        policy = self._agent_policy(agent_id)
        if policy is None or policy.get("deny_all_tools", False):
            return []
        return [str(name) for name in policy.get("allowed_tools", [])]

    def filter_tool_definitions(
        self,
        agent_id: str,
        tools: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Filter OpenAI-format tool definitions before they reach the model.

        Execution-time validation remains mandatory; this method additionally
        prevents a prompt/schema mismatch from inviting calls that will later
        be denied.
        """

        allowed = set(self.get_allowed_tools(agent_id))
        filtered: List[Dict[str, Any]] = []
        for tool in tools or ():
            function = tool.get("function", {}) if isinstance(tool, Mapping) else {}
            name = function.get("name") if isinstance(function, Mapping) else None
            if name in allowed:
                filtered.append(dict(tool))
        return filtered

    @staticmethod
    def _constraint_names(constraints: Sequence[Any]) -> tuple[set[str], Dict[str, Any]]:
        names: set[str] = set()
        settings: Dict[str, Any] = {}
        for constraint in constraints:
            if isinstance(constraint, str):
                names.add(constraint)
            elif isinstance(constraint, dict):
                names.update(str(key) for key in constraint)
                settings.update(constraint)
        return names, settings

    @staticmethod
    def _contains_any(text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)

    def _has_care_instruction(self, output: str) -> bool:
        care_terms = self.agent_constraints.get("emergency_triage", {}).get("care_terms", [])
        return self._contains_any(output, care_terms)

    def _contains_unexplained_jargon(self, output: str) -> bool:
        for term in self._JARGON_TERMS:
            start = output.find(term)
            if start < 0:
                continue
            vicinity = output[start : start + len(term) + 24]
            if not self._contains_any(vicinity, ["（", "(", "即", "也就是", "指的是"]):
                return True
        return False

    def validate_output(
        self,
        agent_id: str,
        output: str,
        user_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a worker or final response and describe every violation."""

        canonical = self.normalize_agent_id(agent_id)
        policy = self._agent_policy(agent_id)
        unknown_agent = policy is None
        if policy is None:
            # Unknown agents get the strict final-output policy rather than an
            # unconstrained empty policy.  The unknown identity is still a
            # blocking violation below.
            policy = self.agent_constraints["agents"].get("final_answer", {})

        common = self.agent_constraints.get("common", {})
        all_constraints = list(policy.get("output_constraints", [])) + list(
            common.get("output_constraints", [])
        )
        constraint_names, settings = self._constraint_names(all_constraints)
        safety_rules = set(common.get("safety_rules", []))
        blocking_rules = set(common.get("blocking_rules", []))
        forbidden_actions = set(policy.get("forbidden_actions", []))

        text = str(output or "")
        details: List[Dict[str, Any]] = []
        auto_fixable: List[str] = []

        def add(
            code: str,
            message: str,
            *,
            fix: Optional[str] = None,
            force_blocking: Optional[bool] = None,
        ) -> None:
            blocking = code in blocking_rules if force_blocking is None else force_blocking
            details.append(
                {
                    "code": code,
                    "message": message,
                    "severity": "error" if blocking else "warning",
                    "blocking": blocking,
                    "auto_fix": fix,
                }
            )
            if fix:
                auto_fixable.append(fix)

        if unknown_agent:
            add("unknown_agent", f"未知 Agent {agent_id!r} 的输出不能直接发布", force_blocking=True)
        if not text.strip():
            add("empty_output", "回答为空", force_blocking=True)

        if "must_include_disclaimer" in constraint_names:
            if not self._contains_any(text, ["免责声明", "仅供参考", "不能替代专业医生"]):
                add("must_include_disclaimer", "缺少医疗免责声明", fix="add_disclaimer")

        max_length = settings.get("max_response_length")
        if max_length is not None and len(text) > int(max_length):
            add(
                "max_response_length",
                f"回答过长（{len(text)} > {int(max_length)} 字）",
                fix=f"truncate:{int(max_length)}",
            )

        risk_source = str(user_input) if user_input is not None else text
        emergency_terms = self._emergency_matches(risk_source)
        must_triage = bool(emergency_terms) and (
            "must_recommend_doctor_visit_if_high_risk" in constraint_names
            or "always_recommend_doctor_for_serious_symptoms" in safety_rules
        )
        if must_triage and not self._has_care_instruction(text):
            add(
                "must_recommend_doctor_visit_if_high_risk",
                "高危输入没有明确建议立即急诊或拨打急救电话",
                fix="add_emergency_warning",
            )

        delay_terms = self.agent_constraints.get("emergency_triage", {}).get("delay_terms", [])
        if emergency_terms and self._contains_any(text, delay_terms):
            add(
                "never_delay_emergency_care",
                "回答可能延误紧急医疗处置",
                fix="remove_delay_language",
            )

        if "must_use_respectful_tone" in constraint_names and self._contains_any(
            text, self._DISRESPECTFUL_TERMS
        ):
            add("must_use_respectful_tone", "回答包含不尊重用语")

        if "must_explain_reasoning" in constraint_names and not self._contains_any(
            text, ["依据", "原因", "推理", "因为", "支持证据", "需要排除"]
        ):
            add("must_explain_reasoning", "没有解释判断依据")

        if "must_cite_sources" in constraint_names and not any(
            pattern.search(text) for pattern in self._EVIDENCE_MARKERS
        ):
            add("must_cite_sources", "研究型回答没有可核验来源；安全门不会伪造引用")

        if "must_distinguish_facts_from_opinions" in constraint_names:
            fact_marker = self._contains_any(text, ["证据", "研究显示", "指南", "已知事实"])
            uncertainty_marker = self._contains_any(
                text, ["推测", "可能", "观点", "局限", "不确定", "证据不足"]
            )
            if not (fact_marker and uncertainty_marker):
                add("must_distinguish_facts_from_opinions", "未区分证据事实与推测/局限")

        if "no_medical_jargon_without_explanation" in constraint_names:
            if self._contains_unexplained_jargon(text):
                add("no_medical_jargon_without_explanation", "医学术语未作通俗解释")

        if "no_fear_mongering" in constraint_names and self._contains_any(text, self._FEAR_TERMS):
            add("no_fear_mongering", "回答包含恐吓性表述")

        false_hope = self._contains_any(text, self._GUARANTEE_PATTERNS)
        if "no_false_hope" in constraint_names and false_hope:
            add("no_false_hope", "回答包含无法保证的绝对承诺", fix="remove_guarantees")

        privacy_matches = self._privacy_matches(text)
        if (
            "respect_patient_privacy" in constraint_names
            or "respect_patient_privacy" in safety_rules
        ) and privacy_matches:
            add(
                "respect_patient_privacy",
                "回答暴露了可识别个人信息",
                fix="redact_sensitive_data",
            )

        definitive = any(pattern.search(text) for pattern in self._DIAGNOSIS_PATTERNS)
        diagnosis_actions = {"diagnose_disease", "give_definitive_diagnosis", "give_diagnosis"}
        if definitive and forbidden_actions.intersection(diagnosis_actions):
            code = next(iter(forbidden_actions.intersection(diagnosis_actions)))
            add(code, "回答作出了确定性诊断", fix="soften_diagnosis")

        prescription = any(pattern.search(text) for pattern in self._PRESCRIPTION_PATTERNS)
        prescription_actions = {
            "prescribe_medication",
            "recommend_specific_drugs",
            "suggest_self_treatment",
        }
        if prescription and forbidden_actions.intersection(prescription_actions):
            code = next(iter(forbidden_actions.intersection(prescription_actions)))
            add(code, "回答给出了具体处方或自行用药指令", fix="remove_prescription")

        if (
            "recommend_treatment" in forbidden_actions
            and (self._TREATMENT_RECOMMENDATION.search(text) or prescription)
        ):
            add("recommend_treatment", "研究 Agent 越界给出了个体治疗指令")

        if false_hope and "guarantee_cure" in forbidden_actions:
            add("guarantee_cure", "回答承诺治愈或绝对效果", fix="remove_guarantees")

        blocking = any(detail["blocking"] for detail in details)
        blocking_unfixable = [
            detail["code"]
            for detail in details
            if detail["blocking"] and not detail.get("auto_fix")
        ]
        return {
            "valid": not details,
            "blocked": blocking,
            "agent_id": canonical or str(agent_id),
            "violations": [detail["message"] for detail in details],
            "violation_codes": [detail["code"] for detail in details],
            "violation_details": details,
            "auto_fixable": _unique(auto_fixable),
            "blocking_unfixable": _unique(blocking_unfixable),
            "emergency_terms": emergency_terms,
        }

    def enforce_output(
        self,
        agent_id: str,
        output: str,
        user_input: Optional[str] = None,
        auto_fixer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Apply deterministic fixes, revalidate, and release only safe output."""

        triage = self.precheck_input(user_input or "")
        if triage["blocked"]:
            # The generated text is intentionally withheld.  This mirrors the
            # pre-model short circuit and prevents a plausible-sounding answer
            # from distracting from urgent care.
            return {
                "valid": False,
                "gate_passed": False,
                "blocked": True,
                "released": False,
                "output": triage["response"],
                "auto_fixed": [],
                "input_precheck": triage,
                "initial_validation": self.validate_output(
                    agent_id, output, user_input=user_input
                ),
                "final_validation": None,
            }

        initial = self.validate_output(agent_id, output, user_input=user_input)
        fixed_output = str(output or "")
        applied: List[str] = []

        if initial["auto_fixable"]:
            if auto_fixer is None:
                from validation import AutoFixer

                auto_fixer = AutoFixer()
            fixed_output = auto_fixer.fix_output(
                fixed_output,
                initial["auto_fixable"],
                user_input=user_input,
            )
            applied = list(initial["auto_fixable"])

        final = self.validate_output(agent_id, fixed_output, user_input=user_input)
        if final["blocked"]:
            released_output = (
                "当前生成内容未通过医疗安全校验，因此未向你展示。"
                "如有明显不适、症状持续或加重，请及时联系有资质的医生；"
                "如出现紧急症状，请立即拨打 120（或所在地急救电话）。\n\n"
                "【免责声明】以上信息仅供参考，不能替代专业医生的诊断和治疗。"
            )
            released = False
        else:
            released_output = fixed_output
            released = True

        return {
            "valid": not final["violation_details"],
            "gate_passed": not final["blocked"],
            "blocked": final["blocked"],
            "released": released,
            "output": released_output,
            "auto_fixed": applied,
            "input_precheck": triage,
            "initial_validation": initial,
            "final_validation": final,
        }

    gate_output = enforce_output

    def validate_task_decomposition(
        self,
        question: str,
        subtasks: Sequence[Any],
    ) -> Dict[str, Any]:
        """Validate Swarm bounds, identities, dependencies, and required agents."""

        swarm = self.swarm_constraints["swarm"]
        issues: List[str] = []
        recommendations: List[str] = []
        num_subtasks = len(subtasks)

        if num_subtasks > swarm["max_parallel_tasks"]:
            issues.append(
                f"并行子任务过多（{num_subtasks} > {swarm['max_parallel_tasks']}）"
            )

        normalized: List[Dict[str, Any]] = []
        for index, raw in enumerate(subtasks):
            if isinstance(raw, Mapping):
                task = dict(raw)
            else:
                task = {
                    key: getattr(raw, key, None)
                    for key in ("id", "description", "assigned_agent", "assigned_agents", "dependencies")
                }
            normalized.append(task)
            if not str(task.get("description") or "").strip():
                issues.append(f"子任务 {index + 1} 缺少描述")

            assigned = task.get("assigned_agents")
            if assigned is None:
                assigned = [task.get("assigned_agent")] if task.get("assigned_agent") else []
            if len(assigned) > swarm["max_agents_per_task"]:
                issues.append(
                    f"子任务 {task.get('id', index + 1)} 分配了过多 Agent"
                )
            for agent_id in assigned:
                if self.normalize_agent_id(str(agent_id)) not in {
                    "consultation_agent",
                    "diagnostic_agent",
                    "research_agent",
                }:
                    issues.append(f"子任务分配给未知 Worker: {agent_id}")

        task_ids = [str(task.get("id")) for task in normalized if task.get("id")]
        if len(task_ids) != len(set(task_ids)):
            issues.append("子任务 ID 重复")
        known_ids = set(task_ids)
        for task in normalized:
            unresolved = set(task.get("dependencies") or []) - known_ids
            if unresolved:
                issues.append(
                    f"子任务 {task.get('id', '?')} 引用了不存在的依赖: {sorted(unresolved)}"
                )

        for rule in swarm.get("task_decomposition_rules", []):
            keywords = rule.get("keywords") or str(rule.get("pattern", "")).split("|")
            if not any(keyword and keyword in question for keyword in keywords):
                continue
            maximum = rule.get("max_subtasks")
            minimum = rule.get("min_subtasks", 0)
            if maximum is not None and num_subtasks > int(maximum):
                issues.append(f"{rule['name']} 最多允许 {maximum} 个子任务")
                recommendations.append(f"合并为不超过 {maximum} 个子任务")
            if num_subtasks < int(minimum):
                issues.append(f"{rule['name']} 至少需要 {minimum} 个子任务")
            break

        assigned_agents = {
            self.normalize_agent_id(str(agent_id))
            for task in normalized
            for agent_id in (
                task.get("assigned_agents")
                or ([task.get("assigned_agent")] if task.get("assigned_agent") else [])
            )
        }
        missing_required = set(self.get_required_agents(question)) - assigned_agents
        if missing_required:
            issues.append(f"缺少必须参与的 Agent: {sorted(missing_required)}")

        return {
            "valid": not issues,
            "blocked": bool(issues),
            "issues": issues,
            "recommendations": _unique(recommendations),
            "missing_required_agents": sorted(missing_required),
        }

    def get_required_agents(self, question: str) -> List[str]:
        rules = self.swarm_constraints["swarm"].get("agent_selection_rules", [])
        required: List[str] = []
        for rule in rules:
            symptoms = rule.get("if_symptoms", [])
            keywords = rule.get("if_keywords", [])
            if any(term in question for term in symptoms + keywords):
                required.extend(rule.get("must_include", []))
        return _unique(required)

    def get_swarm_limits(self) -> Dict[str, int]:
        """Return the validated resource limits for the runtime coordinator."""

        swarm = self.swarm_constraints["swarm"]
        return {
            "max_agents_per_task": int(swarm["max_agents_per_task"]),
            "max_parallel_tasks": int(swarm["max_parallel_tasks"]),
            "timeout_seconds": int(swarm["timeout_seconds"]),
        }

    def validate_collaboration_mode(self, mode: str) -> Dict[str, Any]:
        """Reject undeclared or explicitly unimplemented collaboration modes."""

        declared = {
            str(item.get("mode")): item
            for item in self.swarm_constraints["swarm"].get("collaboration_modes", [])
        }
        policy = declared.get(str(mode))
        if policy is None:
            return {
                "valid": False,
                "blocked": True,
                "reason": f"未声明的协作模式: {mode}",
            }
        if not bool(policy.get("implemented", False)):
            return {
                "valid": False,
                "blocked": True,
                "reason": f"协作模式 {mode} 尚未实现，不能执行",
            }
        return {"valid": True, "blocked": False, "reason": ""}

    def validate_conflict_resolution(
        self,
        disagreements: Optional[Sequence[Any]] = None,
        resolution: Optional[str] = None,
        *,
        safety_concern: bool = False,
        escalated: bool = False,
    ) -> Dict[str, Any]:
        """Enforce the declared explanation and safety-escalation policy."""

        policy = self.swarm_constraints["swarm"].get("conflict_resolution", {})
        conflicts = list(disagreements or ())
        issues: List[str] = []
        if conflicts and policy.get("require_explanation", False):
            if not str(resolution or "").strip():
                issues.append("Agent 结论存在冲突，但没有记录解决理由")
        if (
            safety_concern
            and policy.get("escalate_if_safety_concern", False)
            and not escalated
        ):
            issues.append("安全相关冲突没有升级到用户/人工审核")
        return {
            "valid": not issues,
            "blocked": bool(issues),
            "strategy": policy.get("strategy", "lead_agent_decides"),
            "issues": issues,
        }

    def validate_swarm_result(
        self,
        contributions: Sequence[Any],
        final_output: str,
        user_input: Optional[str] = None,
        *,
        collaboration_mode: str = "parallel",
        disagreements: Optional[Sequence[Any]] = None,
        resolution: Optional[str] = None,
        safety_concern: bool = False,
        escalated: bool = False,
    ) -> Dict[str, Any]:
        """Validate contribution bounds plus the same final gate used by single mode."""

        quality = self.swarm_constraints["swarm"].get("quality_control", {})
        minimum = int(quality.get("min_contribution_length", 0))
        maximum = int(quality.get("max_contribution_length", 10**9))
        issues: List[str] = []
        rendered_contributions: List[str] = []
        structured_evidence = False

        for index, contribution in enumerate(contributions):
            if isinstance(contribution, Mapping):
                value = contribution.get("result", contribution)
            else:
                value = getattr(contribution, "result", "")
            if isinstance(value, Mapping):
                text = str(value.get("answer") or value.get("content") or value)
                structured_evidence = structured_evidence or any(
                    value.get(key) for key in ("evidence", "sources", "citations", "references")
                )
            else:
                text = str(value)
            rendered_contributions.append(text)
            if len(text) < minimum:
                issues.append(f"第 {index + 1} 个贡献过短")
            if len(text) > maximum:
                issues.append(f"第 {index + 1} 个贡献过长")

        if contributions and quality.get("require_evidence", False):
            textual_evidence = any(
                pattern.search(text)
                for text in rendered_contributions
                for pattern in self._EVIDENCE_MARKERS
            )
            if not structured_evidence and not textual_evidence:
                issues.append("Swarm 贡献没有结构化证据或可核验来源")

        mode_validation = self.validate_collaboration_mode(collaboration_mode)
        if mode_validation["blocked"]:
            issues.append(mode_validation["reason"])
        conflict_validation = self.validate_conflict_resolution(
            disagreements,
            resolution,
            safety_concern=safety_concern,
            escalated=escalated,
        )
        issues.extend(conflict_validation["issues"])

        final_validation = self.validate_output(
            "final_answer", final_output, user_input=user_input
        )
        return {
            "valid": not issues and final_validation["valid"],
            "blocked": bool(issues) or final_validation["blocked"],
            "issues": issues,
            "mode_validation": mode_validation,
            "conflict_validation": conflict_validation,
            "final_validation": final_validation,
        }


class SafetyGate:
    """Small facade for using one safety gate in single and Swarm paths."""

    def __init__(
        self,
        validator: Optional[ConstraintValidator] = None,
        auto_fixer: Optional[Any] = None,
    ):
        self.validator = validator or ConstraintValidator()
        if auto_fixer is None:
            from validation import AutoFixer

            auto_fixer = AutoFixer()
        self.auto_fixer = auto_fixer

    def precheck_input(self, text: str) -> Dict[str, Any]:
        return self.validator.precheck_input(text)

    validate_input = precheck_input

    def finalize_output(
        self,
        output: str,
        agent_id: str = "final_answer",
        user_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.validator.enforce_output(
            agent_id,
            output,
            user_input=user_input,
            auto_fixer=self.auto_fixer,
        )

    gate_output = finalize_output
    apply_final_gate = finalize_output


__all__ = [
    "ConstraintDefinitionError",
    "ConstraintValidator",
    "SafetyGate",
]

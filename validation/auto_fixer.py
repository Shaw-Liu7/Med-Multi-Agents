"""Deterministic repairs for a narrowly defined set of safety violations.

The fixer never invents medical evidence, a diagnosis, or a citation.  Rules
that require new factual content remain unfixable and are blocked by
``constraints.SafetyGate``.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional


logger = logging.getLogger(__name__)


class AutoFixer:
    """Apply auditable, non-generative output repairs."""

    DISCLAIMER = (
        "【免责声明】以上信息仅供参考，不能替代专业医生的诊断和治疗。"
        "如有疑虑或症状持续、加重，请及时就医。"
    )
    EMERGENCY_WARNING = (
        "⚠️【紧急提醒】你描述的情况可能属于医疗急症。请立即拨打 120"
        "（或所在地急救电话）并前往急诊；不要独自驾车，也不要因等待在线回复而延误。"
    )
    _CARE_TERMS = ("立即就医", "急诊", "拨打120", "拨打 120", "急救电话")
    _DELAY_TERMS = (
        "等一等",
        "观察几天",
        "过几天再说",
        "无需就医",
        "不用去医院",
        "在家观察即可",
    )
    _IDENTIFIERS = (
        (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已脱敏]"),
        (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[身份证号已脱敏]"),
        (
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "[邮箱已脱敏]",
        ),
    )
    _PRESCRIPTION_PATTERNS = (
        re.compile(
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?(?:服用|使用|口服|注射)"
            r"[^，。；\n]{0,30}\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|毫升)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?(?:服用|口服|注射)"
            r"[^，。；\n]{1,35}",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:建议|推荐|应当|需要)(?:您|你)?(?:立即)?使用[^，。；\n]{0,20}"
            r"(?:阿司匹林|布洛芬|对乙酰氨基酚|抗生素|激素|胰岛素)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:每日|每天|一日)\s*\d+\s*次[^，。；\n]{0,30}(?:每次)?\s*"
            r"\d+(?:\.\d+)?\s*(?:mg|g|ml|毫克|克|毫升)",
            re.IGNORECASE,
        ),
        re.compile(r"(?:自行|自己)(?:服药|用药|治疗|停药|加量|减量)"),
    )

    def fix_output(
        self,
        output: str,
        auto_fixable: Iterable[str],
        *,
        user_input: Optional[str] = None,
    ) -> str:
        """Apply requested fixes in a safety-preserving order."""

        requested = list(dict.fromkeys(str(item) for item in auto_fixable))
        fixed = str(output or "")

        # First remove or redact unsafe content.
        if "redact_sensitive_data" in requested:
            fixed = self.redact_sensitive_data(fixed)
        if "soften_diagnosis" in requested:
            fixed = self.remove_diagnosis_statements(fixed)
        if "remove_prescription" in requested:
            fixed = self.remove_prescription_statements(fixed)
        if "remove_guarantees" in requested:
            fixed = self.remove_guarantees(fixed)
        if "remove_delay_language" in requested:
            fixed = self.remove_delay_language(fixed)

        # Safety notices are appended after transformations so they cannot be
        # accidentally rewritten by a broad rule.
        if "add_emergency_warning" in requested:
            fixed = self.fix_high_risk_warning(fixed, user_input=user_input)
        if "add_disclaimer" in requested:
            fixed = self.fix_missing_disclaimer(fixed)

        limits: List[int] = []
        for fix in requested:
            if fix.startswith("truncate:"):
                try:
                    limits.append(int(fix.split(":", 1)[1]))
                except ValueError:
                    logger.warning("Ignored malformed truncate repair")
        if limits:
            fixed = self.fix_excessive_length(fixed, min(limits))

        return fixed

    def fix_missing_disclaimer(self, output: str) -> str:
        if any(marker in output for marker in ("免责声明", "仅供参考", "不能替代专业医生")):
            return output
        return f"{output.rstrip()}\n\n{self.DISCLAIMER}".strip()

    def fix_high_risk_warning(
        self,
        output: str,
        *,
        user_input: Optional[str] = None,
    ) -> str:
        del user_input  # The validator, not the fixer, decides whether this repair is needed.
        if any(term in output for term in self._CARE_TERMS):
            return output
        return f"{self.EMERGENCY_WARNING}\n\n{output.lstrip()}".strip()

    def fix_excessive_length(self, output: str, max_length: int) -> str:
        """Truncate prose while preserving emergency and disclaimer blocks."""

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if len(output) <= max_length:
            return output

        mandatory: List[str] = []
        if self.EMERGENCY_WARNING in output:
            mandatory.append(self.EMERGENCY_WARNING)
        if self.DISCLAIMER in output:
            mandatory.append(self.DISCLAIMER)

        body = output
        for block in mandatory:
            body = body.replace(block, "")
        body = body.strip()

        suffix = "\n\n[内容因长度限制已截断]"
        reserved = sum(len(block) + 2 for block in mandatory) + len(suffix)
        body_budget = max(0, max_length - reserved)
        pieces = mandatory + ([body[:body_budget].rstrip() + suffix] if body_budget else [])
        fitted = "\n\n".join(piece for piece in pieces if piece)
        return fitted[:max_length]

    def redact_sensitive_data(self, output: str) -> str:
        redacted = output
        for pattern, replacement in self._IDENTIFIERS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def remove_diagnosis_statements(self, output: str) -> str:
        replacements = {
            "您患有": "这些表现可能与",
            "你患有": "这些表现可能与",
            "您得了": "这些表现可能与",
            "你得了": "这些表现可能与",
            "您就是": "这些表现可能是",
            "你就是": "这些表现可能是",
            "确诊为": "需要由医生面诊和检查后判断是否为",
            "明确诊断为": "需要由医生面诊和检查后判断是否为",
            "最终诊断为": "需要由医生面诊和检查后判断是否为",
            "肯定是": "可能是",
            "一定是": "可能是",
            "百分之百是": "可能是",
        }
        fixed = output
        for original, replacement in replacements.items():
            fixed = fixed.replace(original, replacement)
        return fixed

    def remove_prescription_statements(self, output: str) -> str:
        replacement = "具体药物、剂量和调整方案应由有资质的医生结合面诊结果决定"
        # Replace the entire sentence. Replacing only the matched prefix can
        # leave a dose/frequency fragment behind (for example “每日两次”).
        chunks = re.split(r"([。！？；\n])", output)
        for index in range(0, len(chunks), 2):
            sentence = chunks[index]
            if any(pattern.search(sentence) for pattern in self._PRESCRIPTION_PATTERNS):
                chunks[index] = replacement
        return "".join(chunks)

    def remove_guarantees(self, output: str) -> str:
        replacements = {
            "保证治愈": "治疗效果因人而异",
            "一定能治好": "治疗效果因人而异",
            "百分之百治愈": "无法保证治愈",
            "绝对有效": "效果存在个体差异",
            "不会复发": "仍存在复发可能",
            "完全没有风险": "风险需要结合个体情况评估",
        }
        fixed = output
        for original, replacement in replacements.items():
            fixed = fixed.replace(original, replacement)
        return fixed

    def remove_delay_language(self, output: str) -> str:
        fixed = output
        for term in self._DELAY_TERMS:
            fixed = fixed.replace(term, "请立即就医")
        return fixed


__all__ = ["AutoFixer"]

"""Knowledge provenance helpers used before retrieved text reaches an LLM/user."""

from __future__ import annotations

import re
from typing import Any, Mapping


_CLINICAL_DETAIL = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?\s*(?:mg|g|ml|μg|ug|毫克|克|毫升|单位|片|粒|次/日)"
    r"|(?:每日|每天|一日|每次|每周)\s*\d+"
    r"|(?:口服|注射|静脉|舌下含服|加量|减量|停药|处方|用药剂量)"
    r"|(?:推荐|首选|应当|需要).{0,20}(?:药物|用药|制剂|治疗|手术)"
    r"|(?:抗生素|胰岛素|硝酸甘油|阿司匹林|二甲双胍|药物治疗)"
    r"|(?:≥|≤|>|<)\s*\d+(?:\.\d+)?\s*(?:mmHg|mmol/L|mg/dL|%)?"
    r")",
    re.IGNORECASE,
)


def is_verified(metadata: Mapping[str, Any]) -> bool:
    """Only explicit structured provenance can mark a source as verified."""
    return str(metadata.get("verification_status", "")).lower() == "verified"


def safe_knowledge_excerpt(
    content: str,
    metadata: Mapping[str, Any],
    *,
    max_chars: int = 1_200,
) -> str:
    """Bound retrieved text and suppress unverified dose/threshold instructions.

    This is deliberately conservative. Verification is metadata-driven and cannot
    be inferred from a filename, title, or wording inside the document itself.
    """
    max_chars = max(100, min(int(max_chars), 4_000))
    text = str(content or "").strip()
    if not text:
        return ""
    if is_verified(metadata):
        return text if len(text) <= max_chars else text[:max_chars] + "……[内容已截断]"

    retained = []
    omitted = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _CLINICAL_DETAIL.search(line):
            omitted = True
            continue
        retained.append(line)

    excerpt = "\n".join(retained)
    if omitted:
        excerpt += "\n[未核验资料中的治疗指令、剂量或诊断阈值已省略]"
    if not excerpt:
        excerpt = "[该未核验资料只可作为检索线索，具体内容未展示]"
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars] + "……[内容已截断]"


def provenance_notice(metadata: Mapping[str, Any]) -> str:
    if is_verified(metadata):
        return "来源已在结构化元数据中标记为 verified；仍需确认版本和适用人群。"
    return "本地资料未附可核验出版标识，只能作为检索线索，不能作为处方或现行指南依据。"


__all__ = ["is_verified", "safe_knowledge_excerpt", "provenance_notice"]

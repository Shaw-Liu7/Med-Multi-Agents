"""保守的症状风险分诊 Skill。

规则只负责发现明确红旗信号；未命中规则时返回 ``undetermined``，
绝不把“规则没有覆盖”误报为低风险。
"""

import asyncio
import re
from typing import Any, Dict, List, Tuple

from loguru import logger


EMERGENCY_FLAGS: List[Tuple[str, str]] = [
    ("胸痛", "胸痛可能需要紧急排除心血管事件"),
    ("呼吸困难", "明显呼吸困难属于紧急信号"),
    ("喘不上气", "无法正常呼吸属于紧急信号"),
    ("意识不清", "意识改变属于紧急信号"),
    ("昏迷", "意识丧失属于紧急信号"),
    ("晕厥", "突然晕厥需要紧急评估"),
    ("抽搐", "新发或持续抽搐需要紧急评估"),
    ("面部下垂", "可能存在卒中红旗信号"),
    ("单侧无力", "可能存在卒中红旗信号"),
    ("言语不清", "可能存在卒中红旗信号"),
    ("严重出血", "严重出血需要立即处理"),
    ("大量出血", "大量出血需要立即处理"),
    ("过敏性休克", "严重过敏反应需要立即处理"),
    ("嘴唇发紫", "发绀可能提示严重缺氧"),
    ("自杀", "存在即时人身安全风险"),
    ("自残", "存在即时人身安全风险"),
    ("服药过量", "疑似药物过量需要立即处理"),
]

URGENT_FLAGS: List[Tuple[str, str]] = [
    ("持续呕吐", "持续呕吐可能导致脱水或提示其他急症"),
    ("剧烈腹痛", "剧烈腹痛需要尽快面诊"),
    ("黑便", "消化道出血可能需要尽快评估"),
    ("便血", "出血症状需要尽快评估"),
    ("高热不退", "持续高热需要尽快评估"),
    ("症状迅速加重", "症状快速进展需要尽快评估"),
]

NEGATIONS = ("无", "没有", "未见", "未出现", "否认", "不伴", "并无")
CLAUSE_DELIMITERS = "，,。；;、\n！？!?"
POSITIVE_BREAKERS = ("但", "不过", "同时出现", "伴有", "伴随", "现有", "现在", "出现")


def _is_negated(text: str, keyword_start: int) -> bool:
    """仅在同一分句内传播否定，并识别后续转折/阳性描述。"""
    clause_start = max((text.rfind(mark, 0, keyword_start) for mark in CLAUSE_DELIMITERS), default=-1) + 1
    prefix = text[clause_start:keyword_start]
    negation_matches = [
        (prefix.rfind(term), len(term))
        for term in NEGATIONS
        if prefix.rfind(term) >= 0
    ]
    if not negation_matches:
        return False
    last_negation, negation_length = max(negation_matches, key=lambda item: item[0])
    last_positive_breaker = max(
        (
            prefix.rfind(term)
            for term in POSITIVE_BREAKERS
            if prefix.rfind(term) >= last_negation + negation_length
        ),
        default=-1,
    )
    return last_negation > last_positive_breaker


def _match_flags(text: str, flags: List[Tuple[str, str]]) -> List[str]:
    reasons = []
    for keyword, reason in flags:
        for match in re.finditer(re.escape(keyword), text):
            if not _is_negated(text, match.start()):
                reasons.append(reason)
                break
    return list(dict.fromkeys(reasons))


async def assess_risk(symptoms: str) -> Dict[str, Any]:
    """识别明确红旗并给出分诊动作，不声称完成医学诊断。"""
    symptoms = (symptoms or "").strip()[:2_000]
    if not symptoms:
        return {
            "answer": "请描述症状、开始时间、严重程度，以及是否有胸痛、呼吸困难或意识改变。",
            "risk_level": "undetermined",
            "recommendation": "信息不足，无法分诊。",
            "reasons": [],
            "kb_advice": None,
        }

    logger.info("Assessing symptom risk: input_length={}", len(symptoms))
    emergency_reasons = _match_flags(symptoms, EMERGENCY_FLAGS)
    urgent_reasons = _match_flags(symptoms, URGENT_FLAGS)

    if emergency_reasons:
        risk_level = "emergency"
        reasons = emergency_reasons
        recommendation = "请立即拨打当地急救电话（中国大陆为 120）或前往急诊，不要自行驾车。"
    elif urgent_reasons:
        risk_level = "high"
        reasons = urgent_reasons
        recommendation = "建议今天尽快线下面诊；若症状加重或出现红旗信号，立即急诊。"
    else:
        risk_level = "undetermined"
        reasons = ["当前文本未发现规则覆盖的明确红旗，但这不能证明风险较低"]
        recommendation = (
            "请结合年龄、基础病、持续时间、生命体征和症状趋势咨询医生；"
            "若出现胸痛、呼吸困难、意识改变、单侧无力或明显出血，立即急诊。"
        )

    return {
        "answer": format_assessment(
            symptoms,
            risk_level,
            reasons,
            recommendation,
        ),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "reasons": reasons,
        "kb_advice": None,
        # 分诊热路径不等待向量模型或外部服务，尤其不能延迟急症提示。
        "knowledge_reference": None,
        "rule_version": "triage-v2",
    }


def format_assessment(
    symptoms: str,
    level: str,
    reasons: List[str],
    recommendation: str,
) -> str:
    level_map = {
        "undetermined": "无法仅凭当前信息确定 ⚪",
        "high": "高风险，需尽快面诊 🔴",
        "emergency": "紧急 🚨",
    }
    output = [
        "【症状风险分诊】",
        f"\n症状描述：{symptoms}",
        f"\n分诊结果：{level_map.get(level, level)}",
        "\n判断依据：",
        *(f"  • {reason}" for reason in reasons),
        f"\n行动建议：{recommendation}",
    ]

    if level == "emergency":
        output.append("\n⚠️ 不要等待在线回复；请立即联系急救服务。")
    output.append("\n此结果是保守分诊，不是诊断，也不能排除规则未覆盖的急症。")
    return "\n".join(output)


def assess_risk_sync(symptoms: str) -> Dict[str, Any]:
    return asyncio.run(assess_risk(symptoms))

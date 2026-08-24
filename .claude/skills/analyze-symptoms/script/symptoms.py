"""症状模式整理 Skill。

本模块只做非诊断性的症状归类和信息补全，不根据少量关键词生成疾病清单。
"""

import asyncio
import re
from typing import Any, Dict, List

from loguru import logger
from knowledge.provenance import safe_knowledge_excerpt


_kb_instance = None


def get_knowledge_base():
    """延迟获取知识库，避免模块导入时加载模型。"""
    global _kb_instance
    if _kb_instance is None:
        from knowledge.milvus_kb import MedicalKnowledgeBase

        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


SYMPTOM_CATEGORIES = {
    "respiratory": {
        "keywords": ["咳嗽", "呼吸", "鼻塞", "咽痛", "喉咙", "气短", "咳痰", "胸闷"],
        "name": "呼吸系统",
    },
    "digestive": {
        "keywords": ["腹痛", "腹泻", "恶心", "呕吐", "胃痛", "便秘", "便血", "消化"],
        "name": "消化系统",
    },
    "neurological": {
        "keywords": ["头痛", "头晕", "眩晕", "麻木", "抽搐", "意识", "无力", "言语不清"],
        "name": "神经系统",
    },
    "cardiovascular": {
        "keywords": ["胸痛", "心悸", "心慌", "血压", "晕厥"],
        "name": "心血管系统",
    },
    "musculoskeletal": {
        "keywords": ["关节", "肌肉", "骨骼", "扭伤", "肿胀", "僵硬"],
        "name": "骨骼肌肉系统",
    },
    "general": {
        "keywords": ["发热", "乏力", "消瘦", "水肿", "皮疹", "畏寒"],
        "name": "全身或其他表现",
    },
}


def _split_symptoms(text: str) -> List[str]:
    return [item.strip() for item in re.split(r"[,，;；、\n]+", text) if item.strip()]


async def analyze_symptoms(symptoms: str) -> Dict[str, Any]:
    """整理症状涉及的系统，并提示仍需补充的临床信息。"""
    symptoms = (symptoms or "").strip()[:2_000]
    if not symptoms:
        return {
            "answer": "请描述主要症状、开始时间、持续多久以及是否正在加重。",
            "patterns": [],
            "possible_diseases": [],
            "care_directions": [],
            "kb_insights": [],
        }

    logger.info("Analyzing symptom pattern: input_length={}", len(symptoms))
    symptom_list = _split_symptoms(symptoms) or [symptoms]

    detected = []
    for category_id, category in SYMPTOM_CATEGORIES.items():
        if any(
            keyword in symptom
            for symptom in symptom_list
            for keyword in category["keywords"]
        ):
            detected.append({"id": category_id, "name": category["name"]})

    patterns = []
    if detected:
        patterns.append("症状涉及：" + "、".join(item["name"] for item in detected))
    else:
        patterns.append("仅凭当前描述无法可靠归类，需要补充症状细节")
    if len(detected) > 1:
        patterns.append("症状跨越多个系统，需结合时间顺序、体征和检查综合判断")

    care_directions = [
        "补充起病时间、持续时间、严重程度和变化趋势",
        "补充年龄、基础病、过敏史、当前用药及妊娠可能性（如适用）",
        "说明是否伴随胸痛、呼吸困难、意识改变、单侧无力或明显出血",
    ]

    kb_insights = []
    try:
        kb = await asyncio.to_thread(get_knowledge_base)
        results = await asyncio.to_thread(
            kb.search,
            query=f"{symptoms} 症状特点 就医提示",
            top_k=2,
            filter_type=None,
            min_score=0.2,
        )
        for result in results:
            metadata = result.get("metadata", {})
            kb_insights.append(
                {
                    "info": safe_knowledge_excerpt(
                        result.get("content", ""), metadata, max_chars=300
                    ),
                    "source": metadata.get("source", "医学知识库"),
                    "score": result.get("score", 0.0),
                }
            )
    except Exception as exc:
        logger.warning("Knowledge supplement unavailable: {}", type(exc).__name__)

    return {
        "answer": format_analysis(symptoms, patterns, care_directions, kb_insights),
        "patterns": patterns,
        # 保留旧字段以兼容调用方，但不再输出关键词猜病结果。
        "possible_diseases": [],
        "care_directions": care_directions,
        "kb_insights": kb_insights,
    }


def format_analysis(
    symptoms: str,
    patterns: List[str],
    care_directions: List[str],
    kb_insights: List[Dict[str, Any]] = None,
) -> str:
    """格式化非诊断性症状分析。"""
    output = ["【症状模式整理】", f"\n症状描述：{symptoms}", "\n当前可确认的信息："]
    output.extend(f"  • {pattern}" for pattern in patterns)

    output.append("\n为了更可靠地判断下一步，请补充：")
    output.extend(f"  • {direction}" for direction in care_directions)

    if kb_insights:
        output.append("\n【本地知识库补充】")
        for insight in kb_insights:
            output.append(insight["info"])
            output.append(f"来源：{insight['source']}")

    output.append("\n⚠️ 这只是症状信息整理，不是疾病诊断；紧急程度应以风险分诊为准。")
    return "\n".join(output)


def analyze_symptoms_sync(symptoms: str) -> Dict[str, Any]:
    return asyncio.run(analyze_symptoms(symptoms))

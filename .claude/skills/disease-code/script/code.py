"""
Disease Code Skill
疾病编码查询 Skill（自包含，无需依赖tools）
"""
import asyncio
import re
from typing import Dict, Any
from loguru import logger
from knowledge.provenance import provenance_notice, safe_knowledge_excerpt

# 全局知识库实例
_kb_instance = None

def get_knowledge_base():
    global _kb_instance
    if _kb_instance is None:
        from knowledge.milvus_kb import MedicalKnowledgeBase
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


async def disease_code(disease_name: str) -> Dict[str, Any]:
    """
    查询疾病 ICD-10 编码

    Args:
        disease_name: 疾病名称

    Returns:
        {
            "answer": "格式化的疾病编码信息",
            "icd10_code": "ICD-10 编码",
            "category": "疾病分类"
        }
    """
    disease_name = (disease_name or "").strip()
    if not disease_name:
        return {
            "answer": "疾病名称不能为空。",
            "icd10_code": "",
            "category": "",
            "source": "未检索",
        }
    logger.info("Searching ICD-10 code: disease_name_length={}", len(disease_name))

    # 使用知识库单例
    kb = await asyncio.to_thread(get_knowledge_base)

    # 使用 Milvus 检索疾病编码
    results = await asyncio.to_thread(
        kb.search,
        query=f"{disease_name} ICD-10编码 疾病分类",
        top_k=1,
        filter_type="disease_classification",
        min_score=0.15,
    )

    if results:
        doc = results[0]
        metadata = doc["metadata"]

        # 从内容中提取 ICD-10 编码（简单解析）
        raw_content = doc["content"]
        content = safe_knowledge_excerpt(raw_content, metadata)
        icd10_code = metadata.get("icd10_code", "")
        if not icd10_code:
            match = re.search(r"ICD-?10(?:编码)?\s*[：:]\s*([A-Z][0-9][0-9A-Z.\-]*)", raw_content, re.I)
            if match:
                icd10_code = match.group(1).upper()
        category = metadata.get("category", "")
        if not category:
            category_match = re.search(r"疾病分类\s*[：:]\s*([^\n]+)", raw_content)
            if category_match:
                category = category_match.group(1).strip()

        return {
            "answer": format_code_info(
                disease_name, content, provenance_notice(metadata)
            ),
            "icd10_code": icd10_code,
            "category": category,
            "source": metadata.get("source", "医学知识库"),
            "score": doc["score"],
            "verification_status": metadata.get("verification_status", "unknown"),
            "provenance_notice": provenance_notice(metadata),
        }
    else:
        # 未找到相关内容
        logger.warning(f"No ICD-10 code found in vector DB for {disease_name}")
        return {
            "answer": f"未找到'{disease_name}'的ICD-10编码，建议使用更标准的疾病名称或联系医生咨询。",
            "icd10_code": "",
            "category": "",
            "source": "未找到"
        }


def format_code_info(disease_name: str, content: str, source_notice: str = "") -> str:
    """格式化疾病编码信息"""
    output = [
        f"【疾病编码信息】\n",
        content,
        f"\n来源说明：{source_notice}" if source_notice else "",
    ]

    return "\n".join(item for item in output if item)


def disease_code_sync(disease_name: str) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(disease_code(disease_name))

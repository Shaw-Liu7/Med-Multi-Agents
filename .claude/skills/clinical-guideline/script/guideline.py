"""
Clinical Guideline Skill
临床指南检索 Skill（自包含，无需依赖tools）
"""
import asyncio
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


async def clinical_guideline(query: str, max_results: int = 1) -> Dict[str, Any]:
    """
    检索临床指南

    Args:
        query: 查询内容（疾病名称或治疗主题）
        max_results: 最大结果数（默认1，仅返回最相关的指南）

    Returns:
        {
            "answer": "格式化的临床指南信息",
            "guideline_title": "指南标题",
            "organization": "发布机构"
        }
    """
    query = (query or "").strip()
    max_results = max(1, min(int(max_results), 5))
    if not query:
        return {
            "answer": "指南查询内容不能为空。",
            "guideline_title": "",
            "organization": "",
            "source": "未检索",
        }
    logger.info("Searching clinical guidelines: query_length={}, max_results={}", len(query), max_results)

    # 使用知识库单例
    kb = await asyncio.to_thread(get_knowledge_base)

    # 使用 Milvus 检索临床指南
    results = await asyncio.to_thread(
        kb.search,
        query=f"{query} 临床指南 诊疗规范",
        top_k=max_results,  # 使用传入的 max_results 参数
        filter_type="clinical_guideline",
        min_score=0.15,
    )

    if results:
        doc = results[0]
        metadata = doc["metadata"]

        return {
            "answer": format_guideline(
                safe_knowledge_excerpt(doc["content"], metadata), metadata
            ),
            "guideline_title": metadata.get("title") or f"{query}相关临床指南",
            "organization": metadata.get("organization", "N/A"),
            "year": metadata.get("year", "N/A"),
            "source": metadata.get("source", "医学知识库"),
            "score": doc["score"],
            "verification_status": metadata.get("verification_status", "unknown"),
            "provenance_notice": provenance_notice(metadata),
        }
    else:
        # 未找到相关内容
        logger.warning(f"No clinical guidelines found in vector DB for {query}")
        return {
            "answer": f"未找到'{query}'的相关临床指南，建议使用更具体的疾病名称或联系专业机构获取权威指南。",
            "guideline_title": "",
            "organization": "",
            "source": "未找到"
        }


def format_guideline(content: str, metadata: Dict[str, Any]) -> str:
    """格式化临床指南信息"""
    output = [
        "【临床诊疗指南】\n",
        f"指南名称：{metadata.get('title') or metadata.get('disease', 'N/A')}",
        f"发布机构：{metadata.get('organization', 'N/A')}",
        f"发布年份：{metadata.get('year', 'N/A')}",
        f"核验状态：{metadata.get('verification_status', 'unknown')}",
        f"注意：{provenance_notice(metadata)}",
        f"\n内容：\n{content}"
    ]

    return "\n".join(output)


def clinical_guideline_sync(query: str, max_results: int = 1) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(clinical_guideline(query, max_results))

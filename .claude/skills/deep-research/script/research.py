"""
Deep Research Skill
深度研究 Skill（依赖 RAG 知识库 + Web Search）
整合网络搜索和 Milvus 医学知识库进行深度研究
"""
import asyncio
from typing import Dict, Any
from loguru import logger


async def deep_research(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    """
    深度研究

    Args:
        query: 研究问题
        max_iterations: 最大迭代次数（默认2）

    Returns:
        {
            "answer": "格式化的研究报告",
            "findings": ["发现1", "发现2"],
            "confidence": "high/medium/low"
        }
    """
    query = (query or "").strip()
    if not query:
        return {
            "answer": "请提供需要研究的医学问题。",
            "findings": [],
            "confidence": "low",
            "sources": 0,
            "status": "invalid_input",
        }
    max_iterations = max(1, min(int(max_iterations), 3))
    logger.info(
        "Starting deep research: query_length={}, max_iterations={}",
        len(query),
        max_iterations,
    )

    # 调用深度研究工作流
    import sys
    from pathlib import Path
    # 确保项目根目录在 sys.path 中
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from research.deep_research_workflow import DeepResearchWorkflow

    try:
        workflow = await asyncio.to_thread(DeepResearchWorkflow)
        # max_iterations 表示真正的“检索→综合→质量检查→细化”轮数，
        # 不再偷换成单次搜索结果数。
        report = await workflow.research_with_refinement(
            question=query,
            max_iterations=max_iterations,
        )

        # report 是 ResearchReport 对象，有以下属性：
        # - key_findings: List[str]
        # - evidence_level: str
        # - confidence: float
        # - sources: List[Dict]
        # - summary: str
        # - recommendations: List[str]

        return {
            "answer": format_research_report(query, report),
            "findings": report.key_findings,
            "confidence": "high" if report.confidence > 0.7 else "medium" if report.confidence > 0.4 else "low",
            "sources": len(report.sources),
            "evidence_level": report.evidence_level,
            "status": "completed",
            "data_sources": [source.get("type", "unknown") for source in report.sources],
        }

    except Exception as e:
        logger.error("Deep research failed: {}", type(e).__name__)
        return {
            "answer": "深度研究暂时失败，请稍后重试或使用本地临床指南检索。",
            "findings": [],
            "confidence": "low",
            "sources": 0,
            "status": "error",
            "error_type": type(e).__name__,
        }


def format_research_report(query: str, report) -> str:
    """
    格式化研究报告

    Args:
        query: 原始查询
        report: ResearchReport 对象（来自 evidence_synthesizer.py）
    """
    output = [
        "【深度研究报告】\n",
        f"研究问题：{query}\n"
    ]

    # 关键发现
    if report.key_findings:
        output.append("关键发现：")
        for i, finding in enumerate(report.key_findings, 1):
            output.append(f"{i}. {finding}")
        output.append("")

    # 综合总结
    if report.summary:
        output.append(f"综合分析：\n{report.summary}\n")

    # 证据等级
    output.append(f"证据等级：{report.evidence_level} 级")

    # 置信度
    confidence_percent = f"{report.confidence:.0%}"
    output.append(f"置信度：{confidence_percent}")

    # 信息冲突（如果有）
    if report.conflicts:
        output.append("\n信息冲突：")
        for conflict in report.conflicts:
            output.append(f"- {conflict}")

    # 建议（如果有）
    if report.recommendations:
        output.append("\n建议：")
        for i, rec in enumerate(report.recommendations, 1):
            output.append(f"{i}. {rec}")

    # 来源数量
    if report.sources:
        output.append(f"\n参考来源数量：{len(report.sources)}")

    output.append("\n说明：只统计报告中实际引用的来源；U 级表示来源或研究设计未核验。")

    return "\n".join(output)


def deep_research_sync(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    return asyncio.run(deep_research(query, max_iterations))

"""
深度研究工作流

编排多步骤研究流程：查询规划 → 搜索 → 检索 → 综合 → 验证
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio

from core import LLMClient
from research.web_search import WebSearchTool, SearchResult
from knowledge.milvus_kb import MedicalKnowledgeBase
from research.evidence_synthesizer import EvidenceSynthesizer, ResearchReport

# 全局知识库实例（单例）
_kb_instance = None

def get_knowledge_base():
    """获取知识库单例"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


class DeepResearchWorkflow:
    """
    深度研究工作流

    功能：
    - 多步骤研究流程编排
    - 查询规划和优化
    - 并行搜索和检索
    - 证据综合和质量控制
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        use_web_search: bool = True,
        use_knowledge_base: bool = True
    ):
        """
        初始化工作流

        Args:
            llm_client: LLM 客户端
            use_web_search: 是否使用网络搜索
            use_knowledge_base: 是否使用 Milvus 知识库
        """
        self.llm_client = llm_client or LLMClient()
        self.use_web_search = use_web_search
        self.use_knowledge_base = use_knowledge_base

        # 初始化组件
        self.web_search = WebSearchTool() if use_web_search else None
        # 使用 Milvus 知识库单例（和其他 Skills 共享，避免重复加载模型）
        self.knowledge_base = get_knowledge_base() if use_knowledge_base else None
        self.synthesizer = EvidenceSynthesizer(llm_client=self.llm_client)

    async def run(
        self,
        question: str,
        max_web_results: int = 10,
        max_kb_results: int = 5
    ) -> ResearchReport:
        """
        执行深度研究

        Args:
            question: 研究问题
            max_web_results: 最大网络搜索结果数
            max_kb_results: 最大知识库检索结果数

        Returns:
            研究报告
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("research question cannot be empty")
        max_web_results = max(0, min(int(max_web_results), 30))
        max_kb_results = max(0, min(int(max_kb_results), 20))
        logger.info("Starting DeepResearch: question_length={}", len(question))

        # Step 1: 查询规划
        sub_queries = await self._plan_queries(question)
        logger.info(f"Planned {len(sub_queries)} sub-queries")

        # Step 2: 并行搜索
        web_results: List[SearchResult] = []
        kb_results: List[Dict[str, Any]] = []

        search_tasks = []
        planned_queries = sub_queries[:3] or [question]
        web_per_query = max(1, (max_web_results + len(planned_queries) - 1) // len(planned_queries))
        kb_per_query = max(1, (max_kb_results + len(planned_queries) - 1) // len(planned_queries))

        if max_web_results and self.use_web_search and self.web_search:
            for query in planned_queries:
                search_tasks.append(
                    self._tagged_web_search(query, max_results=web_per_query)
                )

        if max_kb_results and self.use_knowledge_base and self.knowledge_base:
            for query in planned_queries:
                search_tasks.append(
                    self._tagged_kb_search(query, top_k=kb_per_query)
                )

        # 并行执行
        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # 分离结果
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Search task failed: {}", type(result).__name__)
                    continue

                source_type, source_results = result
                if source_type == "web":
                    web_results.extend(source_results)
                elif source_type == "knowledge_base":
                    kb_results.extend(source_results)

        # 同一来源可能被多个子查询命中，在证据综合前去重并限额。
        unique_web: Dict[str, SearchResult] = {}
        for item in web_results:
            unique_web.setdefault(item.url, item)
        web_results = list(unique_web.values())[:max_web_results]

        unique_kb: Dict[str, Dict[str, Any]] = {}
        for item in kb_results:
            key = str(item.get("id") or item.get("metadata", {}).get("doc_id") or item.get("content", ""))
            previous = unique_kb.get(key)
            if previous is None or item.get("score", 0.0) > previous.get("score", 0.0):
                unique_kb[key] = item
        kb_results = sorted(
            unique_kb.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )[:max_kb_results]

        logger.info(f"Collected {len(web_results)} web results, {len(kb_results)} KB results")

        # Step 3: 证据综合
        report = await self.synthesizer.synthesize(
            query=question,
            web_results=web_results,
            kb_results=kb_results
        )
        if not report.key_findings:
            logger.warning("Report has no key findings")

        if not report.summary:
            logger.warning("Report has no summary")

        logger.info("DeepResearch completed")
        return report

    async def _search_milvus(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        从 Milvus 知识库搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            文档列表（字典格式）
        """
        try:
            # Milvus Lite 和 embedding 推理都是同步调用，不阻塞事件循环。
            results = await asyncio.to_thread(
                self.knowledge_base.search,
                query,
                top_k,
                None,
            )
            logger.debug("Milvus search returned {} results", len(results))
            return results
        except Exception as e:
            logger.error("Milvus search failed: {}", type(e).__name__)
            return []

    async def _tagged_web_search(
        self,
        query: str,
        max_results: int,
    ) -> tuple[str, List[SearchResult]]:
        results = await self.web_search.search(query, max_results=max_results)
        return "web", results

    async def _tagged_kb_search(
        self,
        query: str,
        top_k: int,
    ) -> tuple[str, List[Dict[str, Any]]]:
        results = await self._search_milvus(query, top_k=top_k)
        return "knowledge_base", results

    async def _plan_queries(self, question: str) -> List[str]:
        """
        查询规划：将复杂问题拆解为多个子查询

        Args:
            question: 原始问题

        Returns:
            子查询列表
        """
        prompt = f"""你是医学研究助手。请将以下问题拆解为 2-3 个更具体的子查询，以便进行深度研究。

原始问题：{question}

要求：
1. 每个子查询应该聚焦一个特定方面
2. 子查询应该互补，覆盖问题的不同角度
3. 子查询应该简洁明确

输出格式：
每行一个子查询，不需要编号。

示例：
原始问题：2型糖尿病如何治疗？
子查询1：2型糖尿病的药物治疗方案
子查询2：2型糖尿病的生活方式管理
子查询3：2型糖尿病的并发症预防
"""

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])

            # 解析子查询
            lines = response.strip().split('\n')
            sub_queries = []

            for line in lines:
                line = line.strip()
                # 移除可能的编号
                line = line.lstrip('0123456789.-:：）) ')
                if line and len(line) > 5:  # 过滤太短的行
                    sub_queries.append(line)

            # 至少包含原始问题
            if not sub_queries:
                sub_queries = [question]

            # 限制数量
            sub_queries = sub_queries[:3]

            return sub_queries

        except Exception as e:
            logger.error("Query planning failed: {}", type(e).__name__)
            # 降级：返回原始问题
            return [question]

    async def research_with_refinement(
        self,
        question: str,
        max_iterations: int = 2
    ) -> ResearchReport:
        """
        带细化的研究（多轮迭代）

        Args:
            question: 研究问题
            max_iterations: 最大迭代次数

        Returns:
            最终研究报告
        """
        max_iterations = max(1, min(int(max_iterations), 3))
        logger.info(f"Starting iterative research (max_iterations={max_iterations})")

        report = None

        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # 执行研究
            report = await self.run(question)

            # 检查质量
            if report.confidence >= 0.7 and len(report.key_findings) >= 3:
                logger.info(f"High-quality report achieved in iteration {iteration + 1}")
                break

            # 如果是最后一轮，直接返回
            if iteration == max_iterations - 1:
                break

            # 细化查询（基于当前结果）
            if report.key_findings:
                question = f"{question}（关注：{report.key_findings[0]}）"

        return report


# 便捷函数
async def deep_research(
    question: str,
    use_web: bool = True,
    use_kb: bool = True
) -> ResearchReport:
    """
    快速执行深度研究

    Args:
        question: 研究问题
        use_web: 是否使用网络搜索
        use_kb: 是否使用知识库

    Returns:
        研究报告
    """
    workflow = await asyncio.to_thread(
        DeepResearchWorkflow,
        use_web_search=use_web,
        use_knowledge_base=use_kb,
    )
    return await workflow.run(question)

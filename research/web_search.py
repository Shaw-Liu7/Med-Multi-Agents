"""
网络搜索模块

提供医学领域的网络搜索能力，基于 DuckDuckGo Search API (DDGS)
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import asyncio
from bs4 import BeautifulSoup
import httpx
from urllib.parse import urlparse

# 尝试导入 ddgs (新包名) 或 duckduckgo_search (旧包名)
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.error("DDGS not available. Install with: pip install ddgs")


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    title: str
    url: str
    snippet: str  # 摘要
    source: str = "web"  # 来源标识


class WebSearchTool:
    """
    网络搜索工具

    功能：
    - 使用 DuckDuckGo 进行网络搜索
    - 专注于医学领域网站
    - 结果去重和质量过滤
    """

    def __init__(self, timeout: int = 30, enforce_medical_domains: bool = True):
        """
        初始化搜索工具

        Args:
            timeout: HTTP 请求超时时间（秒）
        """
        self.timeout = timeout
        self.enforce_medical_domains = enforce_medical_domains

        # 医学领域权威网站白名单
        self.medical_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "mayoclinic.org",
            "webmd.com",
            "who.int",
            "cdc.gov",
            "nih.gov",
            "nhc.gov.cn",
            "chinacdc.cn",
            "uptodate.com",
            "medscape.com",
            "healthline.com",
            "medicalnewstoday.com"
        ]

    async def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "cn-zh",  # 中国区域，更适合中文搜索
        safesearch: str = "on",  # 严格安全搜索
        timelimit: Optional[str] = None,  # 时间限制：'d'(天), 'w'(周), 'm'(月), 'y'(年)
        retry_count: int = 2  # 重试次数
    ) -> List[SearchResult]:
        """
        执行搜索（参考 shanglv 项目的实现）

        Args:
            query: 搜索查询
            max_results: 最大结果数
            region: 地区设置（cn-zh = 中国区域）
            safesearch: 安全搜索级别（on = 严格）
            timelimit: 时间限制
            retry_count: 重试次数

        Returns:
            搜索结果列表
        """
        if not DDGS_AVAILABLE:
            logger.error("DDGS not available, cannot perform web search")
            return []

        query = (query or "").strip()
        max_results = max(0, min(int(max_results), 20))
        retry_count = max(0, min(int(retry_count), 3))
        if not query or max_results == 0:
            return []

        for attempt in range(retry_count + 1):
            try:
                logger.info(
                    "Web searching: attempt={}, query_length={}, max_results={}",
                    attempt + 1,
                    len(query),
                    max_results,
                )

                # 在查询中添加医学相关关键词，提高结果相关性
                enhanced_query = f"{query} 医学" if "医学" not in query else query

                # DDGS 是同步 API，放入线程避免阻塞 asyncio 事件循环。
                search_results = await asyncio.to_thread(
                    self._search_sync,
                    enhanced_query,
                    max_results * 3,
                    region,
                    safesearch,
                    timelimit,
                )

                # 处理搜索结果
                results = []
                seen_urls = set()
                for result in search_results:
                    url = result.get("href", "")
                    normalized_url = url.split("#", 1)[0].rstrip("/")
                    if not normalized_url or normalized_url in seen_urls:
                        continue
                    seen_urls.add(normalized_url)
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=url,
                        snippet=result.get("body", ""),
                        source="web"
                    )
                    results.append(search_result)

                if self.enforce_medical_domains:
                    results = self.filter_by_domain(results)
                results = results[:max_results]

                if results:
                    logger.info("Web search returned {} results", len(results))
                    return results
                else:
                    logger.warning("Web search returned no eligible results")

            except Exception as e:
                logger.warning(
                    "Web search error: attempt={}, error_type={}",
                    attempt + 1,
                    type(e).__name__,
                )

                if attempt < retry_count:
                    # 等待后重试
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"Web search failed after {retry_count + 1} attempts")
                    return []

        return []

    @staticmethod
    def _search_sync(
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
        timelimit: Optional[str],
    ) -> List[Dict[str, Any]]:
        """同步执行 DDGS；仅由 ``asyncio.to_thread`` 调用。"""
        for backend in ("bing", "duckduckgo", "auto"):
            try:
                kwargs = {
                    "max_results": max_results,
                    "safesearch": safesearch,
                    "region": region,
                    "backend": backend,
                }
                if timelimit:
                    kwargs["timelimit"] = timelimit
                raw = DDGS().text(query, **kwargs)
                results = list(raw)
                if results:
                    return results
            except Exception as exc:
                logger.debug("DDGS backend {} failed: {}", backend, type(exc).__name__)
        return []

    def filter_by_domain(
        self,
        results: List[SearchResult],
        allowed_domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        按域名过滤结果

        Args:
            results: 搜索结果
            allowed_domains: 允许的域名列表（默认使用医学域名白名单）

        Returns:
            过滤后的结果
        """
        if allowed_domains is None:
            allowed_domains = self.medical_domains

        normalized_domains = {domain.lower().strip(".") for domain in allowed_domains}
        filtered = []
        for result in results:
            hostname = (urlparse(result.url).hostname or "").lower().strip(".")
            # 仅允许精确域名或其子域，防止
            # ``who.int.attacker.example`` 这类子串绕过。
            if any(
                hostname == domain or hostname.endswith(f".{domain}")
                for domain in normalized_domains
            ):
                filtered.append(result)

        logger.info(f"Filtered {len(filtered)}/{len(results)} results by domain")
        return filtered

    async def fetch_content(
        self,
        url: str,
        max_length: int = 2000
    ) -> Optional[str]:
        """
        抓取网页内容

        Args:
            url: 网页 URL
            max_length: 最大内容长度

        Returns:
            网页文本内容（提取正文）
        """
        parsed = urlparse(url)
        max_length = max(100, min(int(max_length), 10_000))
        if parsed.scheme not in {"http", "https"}:
            logger.warning("Rejected non-HTTP research URL")
            return None
        try:
            if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
                logger.warning("Rejected URL with credentials or non-standard port")
                return None
        except ValueError:
            logger.warning("Rejected malformed research URL")
            return None
        if self.enforce_medical_domains and not self.filter_by_domain(
            [SearchResult(title="", url=url, snippet="")]
        ):
            logger.warning("Rejected non-medical domain: {}", parsed.hostname)
            return None

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
                headers={"User-Agent": "MediXResearch/1.0"},
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(
                        allowed in content_type
                        for allowed in ("text/html", "text/plain", "application/xhtml+xml")
                    ):
                        logger.warning("Rejected unsupported research content type")
                        return None

                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 1_000_000:
                            logger.warning("Research page exceeded size limit")
                            return None
                    encoding = response.encoding or "utf-8"
                    html = bytes(body).decode(encoding, errors="replace")

                # 使用 BeautifulSoup 提取正文
                soup = BeautifulSoup(html, 'html.parser')

                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()

                # 提取文本
                text = soup.get_text()

                # 清理空白字符
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # 限制长度
                if len(text) > max_length:
                    text = text[:max_length] + "..."

                return text

        except Exception as e:
            logger.error(
                "Failed to fetch research content: host={}, error_type={}",
                parsed.hostname,
                type(e).__name__,
            )
            return None

    async def search_with_content(
        self,
        query: str,
        max_results: int = 5,
        fetch_full_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        搜索并获取内容

        Args:
            query: 搜索查询
            max_results: 最大结果数
            fetch_full_content: 是否抓取完整内容

        Returns:
            包含内容的搜索结果
        """
        # 执行搜索
        results = await self.search(query, max_results=max_results)

        # 如果需要，抓取完整内容
        enriched_results = []
        for result in results:
            enriched = {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "full_content": None
            }

            enriched_results.append(enriched)

        if fetch_full_content and enriched_results:
            contents = await asyncio.gather(
                *(self.fetch_content(result.url) for result in results),
                return_exceptions=True,
            )
            for enriched, content in zip(enriched_results, contents):
                enriched["full_content"] = None if isinstance(content, Exception) else content

        return enriched_results


# 便捷函数
async def search_medical_web(
    query: str,
    max_results: int = 10
) -> List[SearchResult]:
    """
    快速搜索医学网络信息

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    tool = WebSearchTool()
    return await tool.search(query, max_results=max_results)

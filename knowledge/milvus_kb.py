"""
医学知识库（Milvus Lite）。

设计原则：
- 数据库路径相对项目固定，不受当前工作目录影响；
- 文档块使用稳定 ID 并 upsert，反复导入不会产生重复数据；
- 文档类型使用独立字段过滤，不在 JSON 字符串中做模糊匹配；
- COSINE 搜索结果保留 Milvus 返回的相似度方向，不再错误地使用 ``1 - score``。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "knowledge" / "data" / "milvus_lite.db"
DEFAULT_COLLECTION_NAME = "medical_knowledge_v2"


class MedicalKnowledgeBase:
    """医学知识库。

    同一 ``(db_path, collection_name, embedding_model)`` 共享一个实例，
    不同配置互不影响，避免原先“第一次初始化锁死所有配置”的问题。
    """

    _instances: Dict[Tuple[str, str, str], "MedicalKnowledgeBase"] = {}
    _instances_lock = threading.RLock()

    def __new__(
        cls,
        db_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        **_: Any,
    ):
        resolved = str(Path(db_path or DEFAULT_DB_PATH).expanduser().resolve())
        key = (resolved, collection_name, embedding_model)
        with cls._instances_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = super().__new__(cls)
                cls._instances[key] = instance
            return instance

    def __init__(
        self,
        db_path: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        local_files_only: Optional[bool] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        with self._instances_lock:
            if getattr(self, "_initialized", False):
                return

            self.db_path = str(Path(db_path or DEFAULT_DB_PATH).expanduser().resolve())
            self.collection_name = collection_name
            self.embedding_model_name = embedding_model
            self._write_lock = threading.RLock()

            if local_files_only is None:
                allow_download = os.getenv("MEDIX_ALLOW_MODEL_DOWNLOAD", "false").lower() in {
                    "1",
                    "true",
                    "yes",
                }
                local_files_only = not allow_download

            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            logger.info("Loading embedding model: {}", embedding_model)
            self.embedding_model = SentenceTransformer(
                embedding_model,
                device="cpu",
                local_files_only=local_files_only,
            )
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

            logger.info("Connecting to Milvus Lite: {}", self.db_path)
            self.milvus_client = MilvusClient(self.db_path)

            if not self.milvus_client.has_collection(collection_name):
                self.milvus_client.create_collection(
                    collection_name=collection_name,
                    dimension=self.embedding_dim,
                    metric_type="COSINE",
                    auto_id=False,
                )

            self._initialized = True

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
        """按字符分块，保证步长为正数且不生成空块。"""
        text = (text or "").strip()
        if not text:
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        overlap = max(0, min(overlap, chunk_size - 1))
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _stable_chunk_id(doc_id: str, chunk_index: int, content: str = "") -> int:
        """生成正数 int64 稳定主键。"""
        # 主键只由文档和块位置决定；内容更新时 upsert 覆盖旧块，
        # 不会因为内容哈希变化而残留一份重复记录。
        digest = hashlib.sha256(f"{doc_id}:{chunk_index}".encode("utf-8")).hexdigest()
        return int(digest[:15], 16)

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 1024,
        overlap: int = 100,
    ) -> int:
        """幂等地添加或更新文档块。"""
        if not documents:
            return 0

        chunks: List[Dict[str, Any]] = []
        current_chunk_counts: Dict[str, int] = {}
        for document in documents:
            doc_id = str(document.get("id", "")).strip()
            content = str(document.get("content", "")).strip()
            if not doc_id or not content:
                logger.warning("Skipping document without id/content")
                continue

            parts = self._chunk_text(content, chunk_size=chunk_size, overlap=overlap)
            current_chunk_counts[doc_id] = len(parts)
            for index, part in enumerate(parts):
                metadata = dict(document.get("metadata") or {})
                metadata.update(
                    {
                        "doc_id": doc_id,
                        "chunk_id": index,
                        "total_chunks": len(parts),
                    }
                )
                chunks.append(
                    {
                        "id": self._stable_chunk_id(doc_id, index, part),
                        "content": part,
                        "metadata": metadata,
                        "doc_id": doc_id,
                        "doc_type": str(metadata.get("type", "general")),
                        "source": str(metadata.get("source", "")),
                        "chunk_index": index,
                        "content_hash": hashlib.sha256(part.encode("utf-8")).hexdigest(),
                    }
                )

        if not chunks:
            return 0

        vectors = self.embedding_model.encode(
            [chunk["content"] for chunk in chunks],
            show_progress_bar=False,
        )
        data = []
        for chunk, vector in zip(chunks, vectors):
            data.append(
                {
                    **{key: value for key, value in chunk.items() if key != "metadata"},
                    "metadata": json.dumps(chunk["metadata"], ensure_ascii=False),
                    "vector": vector.tolist(),
                }
            )

        with self._write_lock:
            self.milvus_client.upsert(
                collection_name=self.collection_name,
                data=data,
            )
            # 如果更新后的文档块数变少，清掉该文档尾部的旧块。
            # 先完成 upsert，再做精确尾块清理，降低写入失败造成数据缺失的风险。
            for doc_id, current_count in current_chunk_counts.items():
                safe_doc_id = doc_id.replace("\\", "\\\\").replace('"', '\\"')
                try:
                    existing = self.milvus_client.query(
                        collection_name=self.collection_name,
                        filter=f'doc_id == "{safe_doc_id}"',
                        output_fields=["id", "chunk_index"],
                    )
                    stale_ids = [
                        row["id"]
                        for row in existing
                        if int(row.get("chunk_index", -1)) >= current_count
                    ]
                    if stale_ids:
                        self.milvus_client.delete(
                            collection_name=self.collection_name,
                            ids=stale_ids,
                        )
                except Exception as exc:
                    # 兼容旧 collection schema；搜索阶段仍会做结果去重。
                    logger.warning(
                        "Unable to remove stale chunks for one document: {}",
                        type(exc).__name__,
                    )
        logger.info("Upserted {} knowledge chunks", len(data))
        return len(data)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索，支持类型和最低相似度过滤。"""
        query = (query or "").strip()
        if not query or top_k <= 0:
            return []

        query_vector = self.embedding_model.encode([query])[0]
        filter_expr = None
        if filter_type:
            safe_type = filter_type.replace("\\", "\\\\").replace('"', '\\"')
            filter_expr = f'doc_type == "{safe_type}"'

        try:
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_vector.tolist()],
                # 兼容旧数据中可能存在的重复块，先多取一些再去重。
                limit=max(top_k, top_k * 3),
                filter=filter_expr,
                output_fields=[
                    "content",
                    "metadata",
                    "doc_id",
                    "doc_type",
                    "source",
                    "chunk_index",
                    "content_hash",
                ],
            )
        except Exception as exc:
            logger.error("Knowledge search failed: {}", exc)
            return []

        documents: List[Dict[str, Any]] = []
        seen_chunks = set()
        for hits in results or []:
            for hit in hits:
                try:
                    entity = hit.get("entity", {})
                    score = float(hit.get("distance", hit.get("score", 0.0)))
                    if min_score is not None and score < min_score:
                        continue
                    metadata_raw = entity.get("metadata", "{}")
                    metadata = (
                        json.loads(metadata_raw)
                        if isinstance(metadata_raw, str)
                        else dict(metadata_raw or {})
                    )
                    dedupe_key = (
                        str(metadata.get("doc_id") or entity.get("doc_id") or ""),
                        str(metadata.get("chunk_id", entity.get("chunk_index", ""))),
                        str(entity.get("content", "")),
                    )
                    if dedupe_key in seen_chunks:
                        continue
                    seen_chunks.add(dedupe_key)
                    documents.append(
                        {
                            "id": hit.get("id"),
                            "content": entity.get("content", ""),
                            "metadata": metadata,
                            "score": score,
                        }
                    )
                    if len(documents) >= top_k:
                        return documents
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning("Skipping malformed Milvus result: {}", exc)

        return documents

    def count_documents(self) -> int:
        """返回当前 collection 的实体数量。"""
        try:
            rows = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["count(*)"],
            )
            if rows:
                return int(rows[0].get("count(*)", 0))
        except Exception as exc:
            logger.warning("Failed to count knowledge chunks: {}", exc)
        return 0

    def delete_collection(self) -> None:
        """删除 collection；仅供显式管理/测试使用。"""
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)

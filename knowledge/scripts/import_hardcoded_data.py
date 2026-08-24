"""
导入医学知识文档到 Milvus 知识库

数据来源：knowledge/data/documents/*.txt
文档分类：
- 01-09: 生活方式建议
- 10-19: ICD-10疾病编码
- 20-29: 临床指南
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from knowledge.milvus_kb import MedicalKnowledgeBase


CATEGORY_BY_PREFIX = {
    "lifestyle": ("lifestyle", "生活方式建议"),
    "symptoms": ("emergency_symptoms", "急症症状规则"),
    "icd10": ("disease_classification", "ICD-10疾病编码"),
    "guideline": ("clinical_guideline", "临床指南"),
}


def load_documents_from_directory(doc_dir: Path) -> list:
    """从 documents 目录加载所有 txt 文件"""
    documents = []
    txt_files = sorted(doc_dir.glob("*.txt"))

    if not txt_files:
        logger.warning(f"No txt files found in {doc_dir}")
        return documents

    logger.info(f"Found {len(txt_files)} txt files in {doc_dir}")

    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding='utf-8')

            # 从文件名推断文档类型
            filename = txt_file.stem  # 例如：01_lifestyle_hypertension
            parts = filename.split('_', 2)

            if len(parts) < 2:
                logger.warning(f"Skipping {txt_file.name}: invalid filename format")
                continue

            doc_type_prefix = parts[1] if len(parts) > 1 else ""
            disease_name = parts[2] if len(parts) > 2 else ""

            # 以文件名中的语义前缀为准，避免 05_symptoms_emergency
            # 被“01-09”的粗粒度数字规则错分为 lifestyle。
            doc_type, corpus_category = CATEGORY_BY_PREFIX.get(
                doc_type_prefix,
                ("general", "通用医学资料"),
            )

            first_line = content.split('\n')[0].strip()
            display_name = first_line or disease_name or filename
            organization_match = re.search(r"发布机构[：:]\s*([^\n]+)", content)
            year_match = re.search(r"发布年份[：:]\s*(\d{4})", content)

            # 构建文档
            doc = {
                "id": f"{doc_type}_{filename}",
                "content": content,
                "metadata": {
                    "type": doc_type,
                    "disease": display_name,
                    "disease_key": disease_name,
                    "title": display_name,
                    # 仓库没有附带可核验的出版链接/DOI，不能把这些示例文本
                    # 冒充为已验证的权威来源。
                    "source": "本地示例医学资料（未核验）",
                    "corpus_category": corpus_category,
                    "verification_status": "unverified",
                    "intended_use": "development_demo",
                    "organization": organization_match.group(1).strip() if organization_match else "",
                    "year": year_match.group(1) if year_match else "",
                    "filename": txt_file.name,
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            }

            documents.append(doc)
            logger.debug(f"Loaded: {txt_file.name} -> type={doc_type}, disease={disease_name}")

        except Exception as e:
            logger.error(f"Error loading {txt_file.name}: {e}")
            continue

    return documents


def extract_documents_by_type(documents: list, doc_type: str) -> list:
    """按类型筛选文档"""
    return [doc for doc in documents if doc["metadata"]["type"] == doc_type]


def main():
    """主函数：加载文档并导入到 Milvus"""
    parser = argparse.ArgumentParser(description="幂等导入本地医学知识文档")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="允许 sentence-transformers 在本地模型不存在时下载；默认禁止",
    )
    args = parser.parse_args()
    logger.info("=" * 70)
    logger.info("开始导入医学知识文档到 Milvus 知识库")
    logger.info("=" * 70)

    # 文档目录
    doc_dir = Path(__file__).parent.parent / "data" / "documents"

    if not doc_dir.exists():
        logger.error(f"Documents directory not found: {doc_dir}")
        logger.error("Please create the directory and add txt files")
        return

    # 加载所有文档
    logger.info(f"\n📚 从目录加载文档: {doc_dir}")
    all_docs = load_documents_from_directory(doc_dir)

    if not all_docs:
        logger.error("No documents loaded. Please add txt files to knowledge/data/documents/")
        return

    # 统计
    lifestyle_docs = extract_documents_by_type(all_docs, "lifestyle")
    icd10_docs = extract_documents_by_type(all_docs, "disease_classification")
    guideline_docs = extract_documents_by_type(all_docs, "clinical_guideline")
    general_docs = extract_documents_by_type(all_docs, "general")
    emergency_docs = extract_documents_by_type(all_docs, "emergency_symptoms")

    logger.info(f"\n✅ 总共加载 {len(all_docs)} 个文档")
    logger.info(f"   - 生活方式建议: {len(lifestyle_docs)}")
    logger.info(f"   - ICD-10编码: {len(icd10_docs)}")
    logger.info(f"   - 临床指南: {len(guideline_docs)}")
    logger.info(f"   - 急症症状: {len(emergency_docs)}")
    logger.info(f"   - 其他: {len(general_docs)}")

    # 创建知识库实例
    kb = MedicalKnowledgeBase(local_files_only=not args.allow_model_download)

    # 导入数据
    logger.info("\n💾 导入到 Milvus...")
    num_added = kb.add_documents(all_docs)

    logger.info("\n" + "=" * 70)
    logger.info(f"🎉 完成！幂等写入 {num_added} 个文档块到知识库")
    logger.info("=" * 70)

    # 测试检索
    logger.info("\n🔍 测试语义检索...")
    test_queries = [
        ("血压高怎么办", "lifestyle"),
        ("糖尿病编码", "disease_classification"),
        ("高血压治疗指南", "clinical_guideline")
    ]

    for query, filter_type in test_queries:
        results = kb.search(query, top_k=1, filter_type=filter_type)
        if results:
            logger.info(f"  ✅ '{query}' → 找到 {len(results)} 个结果")
            logger.info(f"     最相关: {results[0]['metadata'].get('disease', 'N/A')} (相似度: {results[0]['score']:.3f})")
        else:
            logger.warning(f"  ⚠️  '{query}' → 未找到结果")


if __name__ == "__main__":
    main()

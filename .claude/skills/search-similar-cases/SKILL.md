---
name: search-similar-cases
description: Search similar historical cases from long-term memory (Mem0). Use when user asks "有类似的案例吗", "之前有人问过", "相关病例", or when context from past sessions would be helpful.
---

# Search Similar Cases (搜索相似案例)

在用户明确同意长期记忆后，搜索同一租户、同一用户范围内的历史摘要。

## When to Use

- 用户问"有类似的案例吗""之前有人问过这个问题吗"
- 需要参考历史病例或类似问题的处理经验
- 同一用户范围内的跨会话摘要检索

## 调用方式

```bash
/search-similar-cases 高血压患者的饮食建议
```

## 返回格式

```json
{
  "answer": "格式化的相似案例",
  "total_found": 3,
  "query": "高血压患者的饮食建议"
}
```

## Note

This skill must never search across users or tenants. External long-term storage is disabled by default and requires explicit consent. For current-session history, use `/search-history`.

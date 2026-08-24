---
name: analyze-symptoms
description: Organize symptom patterns by body system and identify missing clinical details. This is not a diagnosis or differential-diagnosis generator.
---

# Analyze Symptoms (症状分析)

整理症状涉及的身体系统，并提示还需要补充哪些信息；不根据关键词猜测疾病。

## When to Use

- 用户描述多个症状，需要模式分析
- 需要梳理就诊前应补充的信息
- 评估症状所涉及的身体系统

## 底层实现

- 技术: 非诊断性症状分类规则 + 可选本地知识检索
- 数据源: 本地示例资料（默认标记为未核验）
- 安全边界: 不生成疾病清单，紧急程度必须另行调用 `assess-risk`

## 调用方式

```bash
/analyze-symptoms 头痛,发热,咳嗽
```

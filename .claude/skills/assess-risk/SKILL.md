---
name: assess-risk
description: Conservatively triage explicit red-flag symptoms as emergency/high/undetermined. A rule miss never means low risk.
---

# Assess Risk (风险评估)

评估症状的风险等级，判断是否需要紧急就医。

## When to Use

- 用户描述症状，需要评估严重程度
- 判断是否需要紧急就医
- 保守分诊（紧急/高风险/无法确定）

## 底层实现

- 技术: 本地确定性红旗规则，不依赖网络、向量模型或 LLM
- 安全边界: 未命中规则返回 `undetermined`，绝不自动判定“低危”
- 急症热路径: 直接返回急救提示，不等待知识库或外部服务

## 调用方式

```bash
/assess-risk 胸痛,呼吸困难
```

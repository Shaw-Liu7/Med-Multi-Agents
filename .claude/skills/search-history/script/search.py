"""
Search Conversation History Skill
搜索当前会话历史（短期记忆）
"""
from typing import Dict, Any
from loguru import logger


async def search_history(session_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    搜索当前会话的历史对话

    Args:
        session_id: 会话ID
        limit: 最多返回多少轮对话（默认10轮）

    Returns:
        {
            "answer": "格式化的历史对话",
            "total_messages": 总消息数,
            "session_id": "会话ID"
        }
    """
    logger.info("Conversation history Skill invoked through unscoped fallback")
    # ShortTermMemory 已取消全局单例。真正执行由 AgentLoop 拦截，并绑定
    # 服务端可信的 tenant/user/session 与同一个注入实例；这里绝不自行 new，
    # 否则既会永远读到空历史，也可能形成越权作用域。
    return {
        "answer": "会话历史只能在带可信请求作用域的 AgentLoop 中检索。",
        "total_messages": 0,
        "session_id": session_id,
        "success": False,
        "error": "scoped_runtime_required",
    }


def format_history(messages: list) -> str:
    """
    格式化历史对话

    Args:
        messages: 消息列表

    Returns:
        格式化的字符串
    """
    if not messages:
        return "无历史记录"

    output = ["【当前会话历史】\n"]

    # 按角色分组显示
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")

        if role == "user":
            output.append(f"【轮次 {i // 2 + 1}】")
            output.append(f"用户: {content}")
        elif role == "assistant":
            output.append(f"系统: {content[:200]}..." if len(content) > 200 else f"系统: {content}")
            output.append("")  # 空行分隔

    return "\n".join(output)


# 同步版本
def search_history_sync(session_id: str, limit: int = 10) -> Dict[str, Any]:
    """同步版本的搜索历史"""
    import asyncio
    return asyncio.run(search_history(session_id, limit))


if __name__ == "__main__":
    # 测试
    import asyncio

    test_session_id = "test_session_123"
    result = asyncio.run(search_history(test_session_id))

    print("=" * 70)
    print(f"会话ID: {test_session_id}")
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"总消息数: {result['total_messages']}")

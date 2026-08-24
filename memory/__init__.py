"""医疗助手的隔离、可审计记忆组件。"""

# 短期和长期记忆
from .short_term import (
    ShortTermMemory,
    ConversationHistory,
    MemoryScope,
)
from .long_term import (
    LongTermMemory,
    LongTermMemoryRecord,
)

# Harness Engineering: 熵管理
from .entropy_manager import (
    MemoryEntropyManager
)

# 本地 Markdown 持久化
from .agent_identity import (
    AgentIdentity,
    AgentIdentityManager,
    CollaborationRecord,
    ToolUsageStats
)
from .session_summary import (
    SessionSummary,
    SessionSummaryManager,
    AgentParticipation,
    KeyFinding,
    Lesson,
    PerformanceMetrics
)

__all__ = [
    # 短期和长期记忆
    'ShortTermMemory',
    'ConversationHistory',
    'MemoryScope',
    'LongTermMemory',
    'LongTermMemoryRecord',
    # Harness Engineering: 熵管理
    'MemoryEntropyManager',
    # 本地持久化类
    'AgentIdentity',
    'AgentIdentityManager',
    'CollaborationRecord',
    'ToolUsageStats',
    'SessionSummary',
    'SessionSummaryManager',
    'AgentParticipation',
    'KeyFinding',
    'Lesson',
    'PerformanceMetrics',
]

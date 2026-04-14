"""
Agent 模块 —— 使用 LangChain 最新的 create_agent 构建智能客服
注意：create_react_agent 已废弃！现在统一使用 from langchain.agents import create_agent
"""
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from tools import ALL_TOOLS
from config import OPENAI_API_KEY, OPENAI_MODEL

# ==========================================
# 1. 系统提示词（定义 Agent 的"人设"）
# ==========================================
SYSTEM_PROMPT = """你是一个专业、友善的 AI 客服助手。请遵循以下规则：

## 你的身份
- 你是「智能小助手」，一家电商平台的在线客服
- 说话语气要亲切、专业，适当使用 emoji

## 工作流程
1. 先理解用户的问题
2. 如果是业务问题（退货、支付、配送等），先搜索知识库
3. 如果用户提供了订单号，帮他查询订单状态
4. 如果问题复杂无法解决，主动创建工单
5. 如果用户说"转人工"，立即转接

## 重要规则
- 只回答与客服相关的问题，其他话题礼貌拒绝
- 回答要简洁明了，不要长篇大论
- 如果不确定答案，诚实说"我不确定"，并建议转人工
- 不要编造信息，一切以知识库和系统数据为准
"""

# ==========================================
# 2. 创建 Agent（最新 API）
# ==========================================
# MemorySaver：让 Agent 记住对话历史（多轮对话的关键）
memory = MemorySaver()

agent = create_agent(
    # model 支持字符串格式 "provider:model_name"，内部自动调用 init_chat_model
    model=f"openai:{OPENAI_MODEL}",
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,    # 注意：新 API 用 system_prompt，不是 prompt
    checkpointer=memory,
    name="customer_service_agent",
)


# ==========================================
# 3. 对外提供的调用函数
# ==========================================
def chat(user_message: str, session_id: str = "default") -> str:
    """
    与 Agent 对话的入口函数

    参数:
        user_message: 用户发的消息
        session_id:   会话ID（不同用户用不同ID，Agent 会分别记住各自的对话）

    返回:
        Agent 的回复文本
    """
    config = {"configurable": {"thread_id": session_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )

    # 提取最后一条 AI 消息作为回复
    ai_message = result["messages"][-1]
    return ai_message.content
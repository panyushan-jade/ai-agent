"""
工具模块 —— Agent 可以调用的外部能力
每个工具就是一个 Python 函数，加上 @tool 装饰器让 Agent 知道它的存在
"""
import random
from langchain_core.tools import tool
from knowledge import KnowledgeBase

# 全局知识库实例
kb = KnowledgeBase()


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库，查找与用户问题相关的业务信息。
    当用户询问退货政策、支付方式、配送时间等业务问题时使用此工具。
    """
    results = kb.search(query, top_k=3)
    if not results:
        return "知识库中未找到相关信息。"

    context = "\n---\n".join([
        f"[来源: {r['source']}]\n{r['content']}"
        for r in results
    ])
    return f"以下是从知识库中检索到的相关信息：\n\n{context}"


@tool
def query_order_status(order_id: str) -> str:
    """
    查询订单状态。当用户提供订单号想查看物流或订单进度时使用此工具。
    参数 order_id: 用户提供的订单号
    """
    # 🔧 实际项目中，这里会调用你的业务 API / 数据库
    # 这里用模拟数据演示
    mock_orders = {
        "ORD001": {"status": "已发货", "logistics": "顺丰快递 SF1234567", "eta": "预计明天送达"},
        "ORD002": {"status": "待发货", "logistics": "暂无", "eta": "预计后天发出"},
        "ORD003": {"status": "已签收", "logistics": "中通快递 ZT9876543", "eta": "已于昨天签收"},
    }

    order = mock_orders.get(order_id.upper())
    if order:
        return (
            f"📦 订单 {order_id} 状态：\n"
            f"  - 状态：{order['status']}\n"
            f"  - 物流：{order['logistics']}\n"
            f"  - 预计：{order['eta']}"
        )
    return f"❌ 未找到订单号 {order_id}，请确认订单号是否正确。"


@tool
def create_ticket(issue_summary: str) -> str:
    """
    创建客服工单。当用户的问题无法立即解决，需要升级处理时使用此工具。
    参数 issue_summary: 问题摘要
    """
    # 🔧 实际项目中，这里调用工单系统 API
    ticket_id = f"TK{random.randint(10000, 99999)}"
    return (
        f"✅ 已为您创建工单：\n"
        f"  - 工单号：{ticket_id}\n"
        f"  - 问题：{issue_summary}\n"
        f"  - 状态：待处理\n"
        f"我们的专员会在24小时内联系您。"
    )


@tool
def transfer_to_human() -> str:
    """
    转接人工客服。当用户明确要求转人工，或 AI 无法解决问题时使用此工具。
    """
    # 🔧 实际项目中，这里会推送到人工客服队列
    return "🙋 正在为您转接人工客服，请稍候...当前排队人数：2人，预计等待时间：3分钟。"


# 所有工具汇总（后面 Agent 会用到��
ALL_TOOLS = [
    search_knowledge_base,
    query_order_status,
    create_ticket,
    transfer_to_human,
]
"""
主服务 —— FastAPI + WebSocket 实时聊天
"""
import os
import uuid
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent import chat
from knowledge import KnowledgeBase

# ==========================================
# 初始化
# ==========================================
app = FastAPI(title="AI 客服 Agent", version="1.0.0")

# 挂载静态文件目录（前端页面）
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup():
    """服务启动时，加载知识库文档"""
    kb = KnowledgeBase()
    kb.load_documents("./docs")
    print("🚀 AI 客服 Agent 已启动！")


# ==========================================
# HTTP 接口
# ==========================================
@app.get("/")
async def index():
    """返回聊天前端页面"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "AI Customer Service Agent"}


# ==========================================
# WebSocket 聊天接口（核心）
# ==========================================
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket 聊天端点
    每个连接生成一个唯一 session_id，Agent 会记住这个会话的上下文
    """
    await websocket.accept()

    # 每个连接一个唯一的会话 ID
    session_id = str(uuid.uuid4())
    print(f"📱 新用户连接，会话ID: {session_id}")

    # 发送欢迎消息
    await websocket.send_json({
        "type": "message",
        "content": "👋 您好！我是智能小助手，有什么可以帮您的吗？\n\n"
                   "我可以帮您：\n"
                   "• 查询订单状态（请提供订单号）\n"
                   "• 了解退货/退款政策\n"
                   "• 了解支付方式和配送信息\n"
                   "• 转接人工客服（输入「转人工」）",
    })

    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()

            if not user_message:
                continue

            # 告诉前端"正在思考"
            await websocket.send_json({
                "type": "thinking",
                "content": "正在思考中...",
            })

            try:
                # 🧠 调用 Agent 获取回复
                reply = chat(user_message, session_id)

                # 返回 Agent 的回复
                await websocket.send_json({
                    "type": "message",
                    "content": reply,
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"抱歉，我遇到了一些问题：{str(e)}",
                })

    except WebSocketDisconnect:
        print(f"👋 用户断开连接，会话ID: {session_id}")


# ==========================================
# 启动入口
# ==========================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
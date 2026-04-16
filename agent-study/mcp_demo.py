# 1. 导入所需模块
import os
import asyncio
from dotenv import load_dotenv

# LangChain 核心组件
from langchain.agents import create_agent  # 新版Agent创建方式
from langgraph.checkpoint.memory import MemorySaver  # 对话记忆

# 向量数据库与嵌入模型
from langchain_openai import OpenAIEmbeddings

# 模型与工具
from langchain_openai import ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool

# MCP 客户端，用于连接高德地图 MCP 服务
from langchain_mcp_adapters.client import MultiServerMCPClient

# 知识库管理
from knowledge import load_or_rebuild_vectorstore

print("1. 正在加载环境变量...")
# 加载 .env 文件中的环境变量
load_dotenv()

# 2. 初始化核心组件
print("2. 正在初始化 LLM 和 嵌入模型...")
# 初始化大语言模型，temperature=0.3 让回答更稳定，适合客服场景
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
llm = ChatOpenAI(
    model="qwen3.6-plus",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 初始化文本嵌入模型，用于将知识库文本转换为向量（使用DashScope兼容接口）
embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
)

# 3. 构建知识库 (RAG - 检索增强生成)
print("3. 正在加载并处理知识库文档...")
vectorstore = load_or_rebuild_vectorstore(embeddings)

# 将检索器 (Retriever) 封装成一个工具，让 Agent 可以调用它来查询知识库
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description="用于搜索公司内部知识库，查找产品信息、公司政策、常见问题等。",
)


# 4. 通过 MCP 连接高德地图服务，获取天气、地图等工具
print("4. 正在连接高德地图 MCP 服务...")


async def load_mcp_tools():
    mcp_client = MultiServerMCPClient(
        {
            "amap-maps": {
                "url": "https://mcp.amap.com/mcp?key=6b6db5c2c79e9dc50fb94a4bbe7021a9",
                "transport": "streamable_http",
            }
        }
    )
    return await mcp_client.get_tools()


mcp_tools = asyncio.run(load_mcp_tools())
print(f"   已加载 {len(mcp_tools)} 个 MCP 工具: {[t.name for t in mcp_tools]}")

# 5. 创建智能客服 Agent（使用 MemorySaver 管理对话记忆）
print("5. 正在创建智能客服 Agent...")
memory = MemorySaver()
agent = create_agent(
    model=llm,
    tools=[retriever_tool] + mcp_tools,  # 知识库检索 + 高德地图 MCP 工具
    checkpointer=memory,
    system_prompt="""你是一个专业的、乐于助人的智能客服助手。
    你的主要职责是回答用户的问题，解决问题，并提供帮助。
    在回答问题时，请优先使用"search_knowledge_base"工具查询公司内部知识库。
    当用户询问天气、地图、路线等信息时，请使用高德地图相关工具。
    如果知识库中没有相关信息，或者问题超出你的知识范围，请礼貌地告知用户，并建议他们联系人工客服。""",
)

# 6. 启动交互式对话
print("\n智能客服已上线！输入 'quit' 或 'exit' 退出对话。")
print("-" * 50)


async def chat_loop():
    while True:
        user_input = input("用户: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 流式输出，逐 token 打印避免卡住控制台
        print("客服: ", end="", flush=True)
        async for msg, metadata in agent.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="messages",
            config={"configurable": {"thread_id": "session-1"}},
        ):
            if msg.content and msg.type == "AIMessageChunk":
                print(msg.content, end="", flush=True)
        print(f"\n{'-' * 50}")


asyncio.run(chat_loop())

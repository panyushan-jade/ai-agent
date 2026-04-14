# import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,ChatMessagePromptTemplate,FewShotPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 依然使用 ChatOpenAI 指向百炼端点
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="sk-c58eb673d8b34733b8c8943fd2825563",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 2. 定义工具（一个普通的 Python 函数 + 文档字符串作为描述）
def get_weather(city: str) -> str:
    """查询指定城市的实时天气情况。"""
    # 实际逻辑可调用天气 API，此处为模拟
    return f"{city} 当前晴朗，气温 22°C。"

# 3. 使用 ChatPromptTemplate 构建提示词模板
prompt = ChatPromptTemplate.from_messages([
    SystemMessage('你是一个智能助手，能够回答用户的问题并调用工具获取信息。'),
    ("human", "请告诉我{city}的天气。"),
])

# 4. 创建智能体  create_agent内部会根据工具的描述自动构建调用工具的逻辑，无需手动编写 if-else 语句
agent = create_agent(
    model=llm,
    tools=[get_weather],
)

# 5. 用模板生成 messages，传入 agent
city = input("请输入要查询天气的城市：")
messages = prompt.format_messages(city=city)

# 流式执行（逐 token 输出）
for msg, metadata in agent.stream(
    {"messages": messages}, stream_mode="messages"
):
    if msg.content and msg.type == "AIMessageChunk":
        print(msg.content, end="", flush=True)
print()

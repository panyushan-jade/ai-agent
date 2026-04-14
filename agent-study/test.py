# 1. 导入所需模块
import os
from dotenv import load_dotenv

# LangChain 核心组件
from langchain.agents import create_agent  # 新版Agent创建方式
from langgraph.checkpoint.memory import MemorySaver  # 对话记忆

# 文档加载与处理 (用于RAG)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量数据库与嵌入模型
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 模型与工具
from langchain_openai import ChatOpenAI
from langchain_core.tools.retriever import create_retriever_tool

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
# 加载 './knowledge_base' 目录下的所有 .txt 文件
loader = DirectoryLoader('./knowledge_base', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

# 将长文档分割成小块，便于检索
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 创建向量数据库 (Chroma)，并将分割后的文档块存入
vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

# 将检索器 (Retriever) 封装成一个工具，让 Agent 可以调用它来查询知识库
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever,
    name="search_knowledge_base",
    description="用于搜索公司内部知识库，查找产品信息、公司政策、常见问题等。",
)

# 4. (可选) 定义一个自定义工具，让客服可以查询实时信息，如天气
def get_weather(city: str) -> str:
    """当用户询问某个城市的天气时，调用此工具获取信息。"""
    # 这里是一个模拟实现，实际应用中应接入真实的天气API
    return f"{city} 今天阳光明媚，气温 22°C。"
    # print(f"  -> 调用工具: 查询 {city} 的天气...")

# 5. 创建智能客服 Agent（使用 MemorySaver 管理对话记忆）
print("5. 正在创建智能客服 Agent...")
memory = MemorySaver()
agent = create_agent(
    model=llm, # 使用我们初始化好的模型
    tools=[retriever_tool, get_weather], # 赋予 Agent 知识库检索和天气查询的能力
    checkpointer=memory, # 使用 MemorySaver 自动管理对话历史
    system_prompt="""你是一个专业的、乐于助人的智能客服助手。
    你的主要职责是回答用户的问题，解决问题，并提供帮助。
    在回答问题时，请优先使用“search_knowledge_base”工具查询公司内部知识库。
    如果知识库中没有相关信息，或者问题超出你的知识范围，请礼貌地告知用户，并建议他们联系人工客服。""",
)

# 7. 启动交互式对话
print("\n智能客服已上线！输入 'quit' 或 'exit' 退出对话。")
print("-" * 50)

while True:
    user_input = input("用户: ")
    if user_input.lower() in ["quit", "exit"]:
        break

    # 流式输出，逐 token 打印避免卡住控制台
    print("客服: ", end="", flush=True)
    for msg, metadata in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="messages",
        config={"configurable": {"thread_id": "session-1"}},
    ):
        if msg.content and msg.type == "AIMessageChunk":
            print(msg.content, end="", flush=True)
    print(f"\n{'-' * 50}")
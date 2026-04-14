import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"                    # 最新模型
EMBEDDING_MODEL = "text-embedding-3-small"  # 最新嵌入模型
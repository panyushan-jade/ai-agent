import os
import hashlib
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def get_kb_hash(kb_dir="./knowledge_base"):
    """计算知识库所有文件的内容哈希，用于检测知识库是否变化"""
    h = hashlib.md5()
    for f in sorted(Path(kb_dir).rglob("*.txt")):
        h.update(f.read_bytes())
    return h.hexdigest()


def load_or_rebuild_vectorstore(embeddings, kb_dir="./knowledge_base", db_dir="./chroma_db"):
    """加载或重建向量数据库：知识库无变化时直接加载，有变化时重建"""
    hash_file = os.path.join(db_dir, ".kb_hash")
    current_hash = get_kb_hash(kb_dir)

    if os.path.exists(hash_file) and open(hash_file).read() == current_hash:
        print("   知识库未变化，加载已有向量数据库...")
        return Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
        )

    print("   知识库有更新，重新构建向量数据库...")
    loader = DirectoryLoader(
        kb_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    vs = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=db_dir
    )

    os.makedirs(db_dir, exist_ok=True)
    with open(hash_file, "w") as f:
        f.write(current_hash)

    return vs



# **不多余，而且推荐保留。** 原因：

# ### 不加哈希检测的代价

# 每次运行 `Chroma.from_documents(texts, embedding=embeddings)` 会：

# 1. **调用嵌入 API** — 你用的是 DashScope `text-embedding-v4`，每次启动都会把所有文档重新发送到远程 API 做向量化，**产生费用和网络耗时**
# 2. **重写数据库** — 即使内容一样也全量写入，浪费 I/O

# 知识库越大，这两项开销越明显。哈希检测让启动从"秒级 API 调用"变成"毫秒级本地加载"。

# ### 有没有害？

# 几乎没有。唯一需要注意的边界情况：

# | 场景 | 表现 | 解法 |
# |------|------|------|
# | 换了嵌入模型但知识库文本没变 | 哈希一样，不会重建，向量和新模型不匹配 | 手动删除 `chroma_db/` 目录 |
# | MD5 碰撞 | 理论存在，实际不可能发生 | 无需处理 |

# **结论：方案不多余（省钱省时间），也不有害（唯一边界情况可手动删目录解决）。**
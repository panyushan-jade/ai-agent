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

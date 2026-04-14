"""
知识库模块 —— 使用 ChromaDB + OpenAI Embedding 构建 RAG
通俗理解：给 AI 一本"业务参考书"，回答问题前先翻书找答案
"""
import os
import chromadb
from langchain_openai import OpenAIEmbeddings
# ✅ 新的
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import OPENAI_API_KEY, EMBEDDING_MODEL


class KnowledgeBase:

    def __init__(self, collection_name: str = "customer_service"):
        # ① 初始化 Embedding 模型（把文字变成数字向量，方便计算相似度）
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

        # ② 初始化 ChromaDB（本地向量数据库，数据存到磁盘）
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦相似度
        )

        # ③ 文本切分器（把长文档切成小段，AI 更容易理解）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )

    def load_documents(self, docs_dir: str = "./docs"):
        """
        从 docs/ 文件夹加载所有 .txt 文件，切分后存入向量数据库
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []

        for filename in os.listdir(docs_dir):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 切分文档
            chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}-chunk-{i}")
                all_metadatas.append({"source": filename})

        if not all_chunks:
            print("⚠️  docs/ 文件夹中没有找到 .txt 文件")
            return

        # 生成向量 & 存入 ChromaDB
        vectors = self.embeddings.embed_documents(all_chunks)
        self.collection.upsert(
            documents=all_chunks,
            embeddings=vectors,
            ids=all_ids,
            metadatas=all_metadatas,
        )
        print(f"✅ 已加载 {len(all_chunks)} 个文档片段到知识库")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        根据用户问题，从知识库中检索最相关的文档片段
        通俗理解：用户问了一个问题，我们去"参考书"里找最相关的几段话
        """
        query_vector = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        # 整理成易读的格式
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i],
            })
        return docs
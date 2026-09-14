'''
    向量数据库底层封装
'''

from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

def build_vector_store(
    split_docs: List[Document],
    embedding_model: Embeddings,
    persist_path: str
) -> Chroma:
    """构建并持久化向量数据库"""
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_path
    )
    print("✅ 向量知识库构建完成\n")
    return vector_db

def load_vector_store(
    persist_path: str,
    embedding_model: Embeddings
) -> Chroma:
    """加载已有的向量数据库"""
    return Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model
    )

def get_retriever(vector_db: Chroma, top_k: int = 3):
    """获取向量检索器"""
    return vector_db.as_retriever(search_kwargs={"k": top_k})
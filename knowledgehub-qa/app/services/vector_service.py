'''
向量库业务封装，带缓存机制
'''

import os
from app.config import DEFAULT_CHUNK_THRESHOLD, EMBED_MODEL_NAME, EMBED_DEVICE
from app.core.models import embedding_model
from app.utils.doc_processor import load_folder_documents, semantic_chunk_documents
from app.db.vector_db import build_vector_store, load_vector_store, get_retriever
from app.services.document_service import DocumentService

class VectorService:
    _kb_cache = {}  # 向量库实例缓存

    @classmethod
    def build_kb(cls, kb_id: str) -> dict:
        """构建指定知识库的向量索引"""
        kb_dir = DocumentService.get_kb_path(kb_id)

        documents = load_folder_documents(kb_dir)
        if not documents:
            raise ValueError("该知识库下没有可加载的文档")

        split_docs = semantic_chunk_documents(
            documents,
            EMBED_MODEL_NAME,
            EMBED_DEVICE,
            threshold=DEFAULT_CHUNK_THRESHOLD
        )

        chroma_path = DocumentService.get_chroma_path(kb_id)
        vector_db = build_vector_store(split_docs, embedding_model, chroma_path)

        cls._kb_cache[kb_id] = vector_db
        return {"doc_count": len(documents), "chunk_count": len(split_docs)}

    @classmethod
    def get_vector_db(cls, kb_id: str):
        """获取向量库实例，优先走缓存"""
        if kb_id in cls._kb_cache:
            return cls._kb_cache[kb_id]

        chroma_path = DocumentService.get_chroma_path(kb_id)
        if not os.path.exists(chroma_path):
            raise ValueError("知识库不存在，请先构建")

        vector_db = load_vector_store(chroma_path, embedding_model)
        cls._kb_cache[kb_id] = vector_db
        return vector_db

    @classmethod
    def get_retriever(cls, kb_id: str, top_k: int = 10):
        """获取知识库检索器"""
        vector_db = cls.get_vector_db(kb_id)
        return get_retriever(vector_db, top_k=top_k)

'''
文档管理业务封装
'''

import os
from typing import List, Tuple
from fastapi import UploadFile

from app.config import KNOWLEDGE_BASE_DIR

class DocumentService:
    @staticmethod
    def get_kb_path(kb_id: str) -> str:
        """获取知识库文件目录路径"""
        return os.path.join(KNOWLEDGE_BASE_DIR, kb_id)

    @staticmethod
    def get_chroma_path(kb_id: str) -> str:
        """获取知识库向量库存储路径"""
        return os.path.join(DocumentService.get_kb_path(kb_id), "chroma_db")

    @staticmethod
    def upload_files(kb_id: str, files: List[UploadFile]) -> Tuple[List[str], List[dict]]:
        """上传文件到指定知识库，返回(成功列表, 失败列表)"""
        kb_dir = DocumentService.get_kb_path(kb_id)
        os.makedirs(kb_dir, exist_ok=True)

        saved_files = []
        failed_files = []

        for file in files:
            try:
                file_path = os.path.join(kb_dir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(file.file.read())
                saved_files.append(file.filename)
            except Exception as e:
                failed_files.append({"filename": file.filename, "error": str(e)})

        return saved_files, failed_files

    @staticmethod
    def list_kb() -> List[dict]:
        """获取所有知识库列表与状态"""
        kb_list = []
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            return kb_list

        for name in os.listdir(KNOWLEDGE_BASE_DIR):
            kb_path = DocumentService.get_kb_path(name)
            if not os.path.isdir(kb_path):
                continue

            file_count = len([
                f for f in os.listdir(kb_path)
                if os.path.isfile(os.path.join(kb_path, f))
            ])
            has_db = os.path.exists(DocumentService.get_chroma_path(name))

            kb_list.append({
                "kb_id": name,
                "file_count": file_count,
                "built": has_db
            })

        return kb_list
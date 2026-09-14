'''
知识库管理接口
'''

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.services.document_service import DocumentService
from app.services.vector_service import VectorService

router = APIRouter(prefix="/api/knowledge", tags=["知识库管理"])

@router.post("/{kb_id}/upload", summary="上传文档到指定知识库")
def upload_document(kb_id: str, files: List[UploadFile] = File(...)):
    saved_files, failed_files = DocumentService.upload_files(kb_id, files)
    return {
        "kb_id": kb_id,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
        "failed_count": len(failed_files),
        "failed_files": failed_files
    }

@router.post("/{kb_id}/build", summary="构建/重建指定知识库的向量索引")
def build_knowledge_base(kb_id: str):
    try:
        result = VectorService.build_kb(kb_id)
        return {"status": "success", "kb_id": kb_id, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败：{str(e)}")

@router.get("/list", summary="获取所有知识库列表")
def list_knowledge_bases():
    return {"knowledge_bases": DocumentService.list_kb()}


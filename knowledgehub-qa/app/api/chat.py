'''
问答交互接口
'''

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.model import ModelConfig
from app.core.rag_engine import rag_answer, init_llm
from app.core.models import reranker
from app.services.vector_service import VectorService

router = APIRouter(prefix="/api/chat", tags=["问答交互"])

class ChatRequest(BaseModel):
    kb_id: str
    question: str
    llm_config: ModelConfig
    use_rerank: bool = True
    recall_top_k: int = 10
    rerank_top_k: int = 3

@router.post("", summary="基于指定知识库问答")
def chat(request: ChatRequest):
    # 加载知识库与检索器
    try:
        retriever = VectorService.get_retriever(
            kb_id=request.kb_id,
            top_k=request.recall_top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载知识库失败：{str(e)}")

    # 初始化大模型
    try:
        llm = init_llm(
            request.llm_config.model_name,
            request.llm_config.api_key,
            request.llm_config.base_url
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"初始化模型失败：{str(e)}")

    # 执行 RAG 问答
    try:
        relevant_docs, answer = rag_answer(
            retriever,
            llm,
            request.question,
            reranker=reranker if request.use_rerank else None,
            rerank_top_k=request.rerank_top_k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成回答失败：{str(e)}")

    # 格式化返回结果
    references = [
        {
            "source": doc.metadata.get("source", "未知"),
            "content": doc.page_content.strip()
        }
        for doc in relevant_docs
    ]

    return {"answer": answer, "references": references}

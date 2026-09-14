'''
模型连通性接口
'''

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.rag_engine import init_llm

router = APIRouter(prefix="/api/model", tags=["模型管理"])

class ModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: str

@router.post("/test", summary="测试大模型配置是否可用")
def test_model(config: ModelConfig):
    try:
        llm = init_llm(config.model_name, config.api_key, config.base_url)
        llm.invoke("ping")
        return {"status": "success", "message": "模型连接成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型连接失败：{str(e)}")

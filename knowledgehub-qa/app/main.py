import os

# 环境变量必须放在最前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.model import router as model_router
from app.api.knowledge import router as knowledge_router
from app.api.chat import router as chat_router
from app.config import DEFAULT_HOST, DEFAULT_PORT

# 预加载全局模型（仅执行一次）
from app.core import models  # noqa: F401

def create_app() -> FastAPI:
    app = FastAPI(
        title="自定义知识库问答助手 API",
        version="1.0.0",
        description="支持自定义大模型、私有知识库的 RAG 问答系统"
    )

    # 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册所有路由
    app.include_router(model_router)
    app.include_router(knowledge_router)
    app.include_router(chat_router)

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=True
    )

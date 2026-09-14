'''
全局模型单例，确保嵌入模型、重排模型只加载一次
'''

from app.config import (
    EMBED_MODEL_NAME, EMBED_DEVICE,
    RERANKER_MODEL_NAME, RERANKER_DEVICE
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.core.reranker import BgeReranker

print("正在加载嵌入模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": EMBED_DEVICE}
)
print("✅ 嵌入模型加载完成")

print("正在加载重排模型...")
reranker = BgeReranker(
    model_name=RERANKER_MODEL_NAME,
    device=RERANKER_DEVICE
)
print("✅ 重排模型加载完成")

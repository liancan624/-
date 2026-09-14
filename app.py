'''
    接口层
'''
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from doc_processor import load_folder_documents, semantic_chunk_documents
from vector_db import build_vector_store, load_vector_store, get_retriever
from rag_engine import init_llm, rag_answer
from reranker import BgeReranker

# ========== 全局初始化（只加载一次，全局复用） ==========
# 嵌入模型和重排模型是通用能力，所有知识库共用，只加载一次
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DEVICE = "cpu"

print("正在加载嵌入模型...")
embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE}
)

print("正在加载重排模型...")
reranker = BgeReranker(model_name="BAAI/bge-reranker-v2-m3", device=EMBED_DEVICE)

# 全局知识库存储：key=知识库ID，value=向量库实例
knowledge_bases = {}

# 知识库根目录
KB_ROOT_DIR = "./knowledge_bases"
os.makedirs(KB_ROOT_DIR, exist_ok=True)

# ========== 请求体定义 ==========
class ModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: str

class ChatRequest(BaseModel):
    kb_id: str
    question: str
    llm_config: ModelConfig
    use_rerank: bool = True
    recall_top_k: int = 10
    rerank_top_k: int = 3

# ========== FastAPI 应用初始化 ==========
app = FastAPI(title="自定义知识库问答助手 API")

# 配置跨域，前端可以直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 1. 模型连通性校验接口 ==========
@app.post("/api/model/test", summary="测试大模型配置是否可用")
def test_model(config: ModelConfig):
    try:
        llm = init_llm(config.model_name, config.api_key, config.base_url)
        # 发一条测试信息验证连通性
        response = llm.invoke("ping")
        return {"status": "success", "message": "模型连接成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"模型连接失败：{str(e)}")

# ========== 2. 文档上传接口 ==========
@app.post("/api/knowledge/{kb_id}/upload", summary="上传文档到指定知识库")
def upload_document(kb_id: str, files: List[UploadFile] = File(...)):
    # 创建对应知识库的文件夹
    kb_dir = os.path.join(KB_ROOT_DIR, kb_id)
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

    return {
        "kb_id": kb_id,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
        "failed_count": len(failed_files),
        "failed_files": failed_files
    }

# ========== 3. 知识库构建接口 ==========
@app.post("/api/knowledge/{kb_id}/build", summary="构建/重建指定知识库的向量索引")
def build_knowledge_base(kb_id: str):
    kb_dir = os.path.join(KB_ROOT_DIR, kb_id)

    if not os.path.exists(kb_dir):
        raise HTTPException(status_code=404, detail="知识库不存在")

    try:
        # 1. 加载文档
        documents = load_folder_documents(kb_dir)
        if not documents:
            raise HTTPException(status_code=400, detail="该知识库下没有可加载的文档")

        # 2. 语义分块
        split_docs = semantic_chunk_documents(
            documents,
            EMBED_MODEL_NAME,
            EMBED_DEVICE,
            threshold=85
        )

        # 3. 构建向量库（每个知识库独立存储）
        chroma_path = os.path.join(KB_ROOT_DIR, kb_id, "chroma_db")
        vector_db = build_vector_store(split_docs, embedding_model, chroma_path)

        # 4. 存入全局缓存
        knowledge_bases[kb_id] = vector_db

        return {
            "status": "success",
            "kb_id": kb_id,
            "doc_count": len(documents),
            "chunk_count": len(split_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败：{str(e)}")

# ========== 4. 问答接口 ==========
@app.post("/api/chat", summary="基于指定知识库问答")
def chat(request: ChatRequest):
    # 检查知识库是否存在
    if request.kb_id not in knowledge_bases:
        # 尝试加载本地已有的知识库
        chroma_path = os.path.join(KB_ROOT_DIR, request.kb_id, "chroma_db")
        if os.path.exists(chroma_path):
            try:
                vector_db = load_vector_store(chroma_path, embedding_model)
                knowledge_bases[request.kb_id] = vector_db
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"加载知识库失败：{str(e)}")
        else:
            raise HTTPException(status_code=404, detail="知识库不存在，请先构建")

    vector_db = knowledge_bases[request.kb_id]

    # 初始化大模型（用户自定义）
    try:
        llm = init_llm(
            request.llm_config.model_name,
            request.llm_config.api_key,
            request.llm_config.base_url
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"初始化模型失败：{str(e)}")

    # 创建检索器
    retriever = vector_db.as_retriever(search_kwargs={"k", request.recall_top_k})

    # 调用RAG问答
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

    # 格式化返回引用片段
    references = [
        {
            "source": doc.metadata.get("source", "未知"),
            "content": doc.page_content.strip()
        }
        for doc in relevant_docs
    ]

    return {
        "answer": answer,
        "references": references
    }

# ========== 5. 知识库列表接口 ==========
@app.get("/api/knowledge/list", summary="获取所有知识库列表")
def list_knowledge_bases():
    kb_list = []
    if os.path.exists(KB_ROOT_DIR):
        for name in os.listdir(KB_ROOT_DIR):
            kb_path = os.path.join(KB_ROOT_DIR, name)
            if os.path.isdir(kb_path):
                has_db = os.path.exists(os.path.join((kb_path, "chroma_db")))
                file_count = len([f for f in os.listdir(kb_path) if os.path.isfile(os.path.join(kb_path, f))])
                kb_list.append({
                    "kb_id": name,
                    "file_count": file_count,
                    "built": has_db
                })

    return {"knowledge_bases": kb_list}

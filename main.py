# ========== 全局环境配置（必须放在所有导入之前） ==========
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ========== 导入各模块 ==========
from langchain_huggingface import HuggingFaceEmbeddings
from doc_processor import load_folder_documents, semantic_chunk_documents
from vector_db import build_vector_store, load_vector_store, get_retriever
from rag_engine import init_llm, rag_answer

# ========== 配置参数 ==========
FOLDER_PATH = "./knowledge_base"
CHROMA_PATH = "./chroma_db"

# 嵌入模型配置
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DEVICE = "cpu"
THRESHOLD = 85

# 大模型配置
LLM_MODEL = "glm-5.3"
LLM_API_KEY = "sk-8024a6bae7074f26913428b0ab98e833.xT1BApCXqltUpVBo"
LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# 检索配置
TOP_K = 3


def main():
    print("=" * 50)
    print("RAG 知识库问答系统")
    print("=" * 50)

    # 1. 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE}
    )

    # 2. 构建/加载向量知识库
    # 首次运行或新增资料时执行构建；后续运行可注释构建，直接加载
    print("\n正在处理文档并构建知识库...\n")
    documents = load_folder_documents(FOLDER_PATH)
    if not documents:
        print("❌ 没有找到可加载的文档！")
        return

    split_docs = semantic_chunk_documents(documents, EMBED_MODEL_NAME, EMBED_DEVICE, THRESHOLD)
    vector_db = build_vector_store(split_docs, embedding_model, CHROMA_PATH)

    # 直接加载已有知识库用下面这行
    # vector_db = load_vector_store(CHROMA_PATH, embedding_model)

    # 3. 初始化大模型和检索器
    llm = init_llm(LLM_MODEL, LLM_API_KEY, LLM_BASE_URL)
    retriever = get_retriever(vector_db, TOP_K)

    # 4. 交互循环
    print("RAG 问答系统已启动，输入问题即可查询，输入 q 退出")
    print("-" * 50)

    while True:
        question = input("\n请输入问题：")
        if question.lower() == "q":
            print("退出系统")
            break

        docs, answer = rag_answer(retriever, llm, question)

        print("\n=== 检索到的参考片段 ===")
        for idx, doc in enumerate(docs):
            print(f"[{idx + 1}] 来源：{doc.metadata.get('source', '未知文件')}")
            print(f"{doc.page_content.strip()}\n")

        print("=== AI 回答 ===")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()

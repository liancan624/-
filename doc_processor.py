'''
    封装所有和文档读取、清洗、分块相关的逻辑
'''

import re
import os

from pypdf import PdfReader
from langchain_core.documents import Document

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document as LlamaDocument


# ========== 文本清洗函数 ==========
def clean_pdf_text(text):
    """清洗PDF提取的文本：去除硬换行、多余空格、页码噪声"""
    # 去除单独的数字页码行
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    # 合并段落内的硬换行
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 去除多个连续空格
    text = re.sub(r' +', ' ', text)
    # 去除首尾空白
    return text.strip()

'''自定义中文分句函数，按中文句末标点符号拆分句子'''
def chinese_sentence_splitter(text):
    sentences = re.split(r'(?<=[。！？；])\s*', text)
    return [s.strip() for s in sentences if s.strip()]

# ========== 文件夹批量加载文档 ==========
def load_folder_documents(folder_path):
    documents = []

    # 支持的文件后缀
    supported_extensions = ['.txt', '.md', '.pdf']

    # 遍历文件夹下所有文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # 获取文件后缀
            file_ext = os.path.splitext(file_name)[1].lower()
            file_path = os.path.join(root, file_name)

            # 只处理支持的格式
            try:
                if file_ext in ['.txt', '.md']:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                elif file_ext == '.pdf':
                    reader = PdfReader(file_path)
                    content = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n\n"

                content = clean_pdf_text(content)

                # 创建Document对象，带上文件名作为元数据，方便溯源
                doc = Document(
                    page_content=content.strip(),
                    metadata={"source": file_name, "path": file_path}
                )
                documents.append(doc)
                print(f"已加载：{file_name}")
            except Exception as e:
                print(f"加载失败{file_name}: {e}")

    return documents

def semantic_chunk_documents(documents, embed_model_name, device, threshold):
    # ===== LlamaIndex语义分块 =====
    # 1. 初始化分块器配套的嵌入模型
    llama_embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name,
        device=device
    )

    # 2. 初始化语义分块器
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=llama_embed_model,
        breakpoint_percentile_threshold=threshold,
        buffer_size=1,
        sentence_splitter=chinese_sentence_splitter  # 传入可调用的分句函数，而非字符串
    )

    # 3. 格式转换：LangChain Document → LlamaIndex Document
    llama_docs = [
        LlamaDocument(text=doc.page_content, metadata=doc.metadata)
        for doc in documents
    ]

    # 4. 执行语义分块
    nodes = semantic_splitter.get_nodes_from_documents(llama_docs)

    # 5. 格式转回：LlamaIndex Node → LangChain Document，无缝对接后续向量库流程
    split_docs = [
        Document(page_content=node.text, metadata=node.metadata)
        for node in nodes
    ]

    print(f"文档切分完成，共{len(split_docs)}个文本块")

    return split_docs
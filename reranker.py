'''
    独立封装所有重排逻辑
'''

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class BgeReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device="cpu"):
        '''
        初始化BGE中文重排模型
        :param model_name: 重排模型名称
        :param device: 运行设备
        '''
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[Document], top_k: int =3) -> list[Document]:
        '''
        对检索到的文档按相关性重排序
        :param query: 用户查询问题
        :param documents: 向量检索返回的 LangChain Document 候选列表
        :param top_k: 返回前 k 个最相关的文档
        :return: 按相关性降序排列的 Document 列表
        '''
        if not documents:
            return []

        # 构造 问题——段落 配对，输入重排模型
        paris = [[query, doc.page_content] for doc in documents]

        # 批量计算相关性分数
        scores = self.model.predict(paris)

        # 按分数降序排列，保留原文档元数据
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 返回 Top k 结果
        return [doc for score, doc in scored_docs[:top_k]]


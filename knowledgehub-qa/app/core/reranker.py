'''
    语义重排序核心组件
'''

from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class BgeReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]



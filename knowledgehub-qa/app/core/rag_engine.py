'''
    RAG 问答编排逻辑
'''

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.core.reranker import BgeReranker

def init_llm(model_name: str, api_key: str, base_url: str) -> ChatOpenAI:
    """初始化大模型实例"""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1
    )


def rag_answer(
        retriever,
        llm: ChatOpenAI,
        question: str,
        reranker: Optional[BgeReranker] = None,
        rerank_top_k: int = 3
) -> Tuple[List[Document], str]:
    """RAG 问答主流程：检索 → 重排 → 生成回答"""
    # 候选检索
    docs = retriever.invoke(question)

    # 语义重排
    if reranker and docs:
        docs = reranker.rerank(question, docs, top_k=rerank_top_k)

    # 构建回答
    prompt_template = """
请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请回答“参考资料中未找到相关内容”。
回答要准确、简洁，有条理。

参考资料：
{context}

用户问题：{question}
回答：
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = chain.invoke(question)
    answer = result["result"]
    source_docs = result["source_documents"]

    # 用重排后的文档覆盖返回结果
    if reranker and docs:
        source_docs = docs

    return source_docs, answer

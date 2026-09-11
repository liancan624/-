'''
    封装大模型配置和问答生成逻辑
'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

'''初始化大模型'''
def init_llm(model_name, api_key, base_url, temperature=0):
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
    return llm

# '''RAG 问答核心逻辑：检索 → 拼接上下文 → 生成回答'''
# def rag_answer(retriever, llm, question):
#     # 检索相关文档
#     relevant_docs = retriever.invoke(question)
#
#     # 把检索到的内容拼接成上下文字符串
#     context = "\n\n".join([doc.page_content for doc in relevant_docs])
#
#     # 构建 Prompt 模板
#     prompt = ChatPromptTemplate.from_template("""
#     你是专业的电力行业知识库助手，请严格根据下面的参考资料回答用户的问题。
#     如果参考资料中没有相关内容，请直接回答"根据现有资料无法回答该问题"，禁止编造内容。
#
#     参考资料：{context}
#
#     用户问题：{question}
#     """)
#
#     # 调用大模型生成回答
#     chain = prompt | llm
#     response = chain.invoke({"context": context, "question": question})
#
#     return relevant_docs, response.content

def rag_answer(retriever, llm, question, reranker=None, rerank_top_k=3):
    '''
    RAG问答核心逻辑
    :param retriever: 向量检索器
    :param llm: 大模型对象
    :param question: 用户问题
    :param reranker: 重排器对象，不传则不做重排
    :param rerank_top_k: 重排后返回的文档数量
    '''

    # 第一步：向量粗召回
    relevant_docs = retriever.invoke(question)

    # 第二步：可选重排精筛
    if reranker:
        relevant_docs = reranker.rerank(question, relevant_docs, top_k=rerank_top_k)

    # 第三步：拼接上下文
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 第四步：构建 Prompt 并生成回答
    prompt = ChatPromptTemplate.from_template("""
    你是专业的电力行业知识库助手，请严格根据下面的参考资料回答用户的问题。
    如果参考资料中没有相关内容，请直接回答"根据现有资料无法回答该问题"，禁止编造内容。

    参考资料：{context}

    用户问题：{question}
    """)

    # 调用大模型生成回答
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return relevant_docs, response.content
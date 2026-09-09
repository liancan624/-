'''
    封装向量库的构建、加载、检索操作
'''

from langchain_chroma import Chroma

'''构建向量数据库'''
def build_vector_store(split_docs, embedding_model, persist_path):
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_path
    )
    print("✅ 向量知识库构建完成\n")
    return vector_db

'''加载已有数据库'''
def load_vector_store(persist_path, embedding_model):
    vector_db = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model
    )
    return vector_db

'''获取检索器'''
def get_retriever(vector_db, top_k=3):
    return vector_db.as_retriever(search_kwargs={"k": 3})

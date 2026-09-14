import streamlit as st
import requests

BACKEND_BASE = "http://localhost:8000"

st.set_page_config(page_title="自定义知识库问答助手", layout="wide")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 模型配置")
    model_name = st.text_input("模型名称", value="glm-4-flash")
    api_key = st.text_input("API Key", type="password")
    base_url = st.text_input("接口地址", value="https://open.bigmodel.cn/api/paas/v4/")

    if st.button("测试连接"):
        resp = requests.post(
            f"{BACKEND_BASE}/api/model/test",
            json={"model_name": model_name, "api_key": api_key, "base_url": base_url}
        )
        if resp.status_code == 200:
            st.success("模型连接成功")
        else:
            st.error(f"连接失败：{resp.json()['detail']}")

    st.divider()
    st.header("📚 知识库管理")

    if st.button("刷新知识库列表") or "kb_list" not in st.session_state:
        resp = requests.get(f"{BACKEND_BASE}/api/knowledge/list")
        if resp.status_code == 200:
            st.session_state["kb_list"] = resp.json()["knowledge_bases"]

    kb_list = st.session_state.get("kb_list", [])
    kb_names = [kb["kb_id"] for kb in kb_list]
    current_kb = st.selectbox("当前知识库", options=kb_names)

    new_kb_id = st.text_input("新建知识库ID")
    if st.button("创建知识库") and new_kb_id:
        st.success(f"知识库 {new_kb_id} 已创建，上传文件后即可构建")

    uploaded_files = st.file_uploader(
        "上传文档",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt", "md"]
    )
    if st.button("上传到当前知识库") and current_kb and uploaded_files:
        files = [("files", (f.name, f.read())) for f in uploaded_files]
        resp = requests.post(
            f"{BACKEND_BASE}/api/knowledge/{current_kb}/upload",
            files=files
        )
        result = resp.json()
        st.success(f"成功上传 {result['saved_count']} 个文件")
        st.rerun()

    if st.button("构建当前知识库", type="primary") and current_kb:
        with st.spinner("正在构建知识库..."):
            resp = requests.post(
                f"{BACKEND_BASE}/api/knowledge/{current_kb}/build"
            )
            if resp.status_code == 200:
                result = resp.json()
                st.success(f"构建完成：{result['doc_count']} 个文档，{result['chunk_count']} 个分块")
            else:
                st.error(f"构建失败：{resp.json()['detail']}")

# ========== 主对话区 ==========
st.title("💬 自定义知识库问答助手")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("references"):
            with st.expander("📄 查看引用来源"):
                for idx, ref in enumerate(msg["references"]):
                    st.caption(f"[{idx + 1}] {ref['source']}")
                    st.write(ref["content"])

if question := st.chat_input("输入你的问题..."):
    if not current_kb:
        st.warning("请先选择一个知识库")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在检索并生成回答..."):
                resp = requests.post(
                    f"{BACKEND_BASE}/api/chat",
                    json={
                        "kb_id": current_kb,
                        "question": question,
                        "llm_config": {
                            "model_name": model_name,
                            "api_key": api_key,
                            "base_url": base_url
                        },
                        "use_rerank": True,
                        "recall_top_k": 10,
                        "rerank_top_k": 3
                    }
                )

                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown(result["answer"])

                    with st.expander("📄 查看引用来源"):
                        for idx, ref in enumerate(result["references"]):
                            st.caption(f"[{idx + 1}] {ref['source']}")
                            st.write(ref["content"])

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "references": result["references"]
                    })
                else:
                    st.error(f"生成失败：{resp.json()['detail']}")
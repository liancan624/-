# RAG问答助手练手项目
> 本项目为私有化RAG问答助手练手工程，代码由AI辅助生成，会持续迭代优化。

## 项目简介
- 目标：搭建可本地部署的私有化问答助手
- 开发方式：基于AI辅助编写代码，有新的优化思路就迭代升级

## 后端服务
项目基于 LangChain 实现基础 RAG 流程，其中使用 LlmaIndex 框架实现语义分块

## 封装接口
项目使用 FastAPI 封装后端接口（app.py文件），在项目根目录执行下面命令启动服务：
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 前端页面
项目使用 Streamlit 快速搭建前端，MVP 阶段快速验证（frontend.py文件），在项目根目录执行下面命令，浏览器会自动打开页面：
```bash
streamlit run frontend.py
```


# RAG问答助手练手项目
> 本项目为私有化RAG问答助手练手工程，代码由AI辅助生成，会持续迭代优化。

## 项目简介
- 目标：搭建可本地部署的私有化问答助手
- 开发方式：基于AI辅助编写代码，有新的优化思路就迭代升级

## 后端服务
项目基于 LangChain 实现基础 RAG 流程，其中使用 LlamaIndex 框架实现语义分块

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

## 更新代码架构
最开始的代码全部堆在根目录，按照分层解耦、单一职责、代码与数据分离的原则，重构文件结构
```
knowledgehub-qa/
├── app/                     # 后端应用主包（所有后端业务代码）
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用总入口（路由注册 + 全局初始化）
│   ├── config.py            # 全局配置中心：所有常量、路径、默认参数统一管理
│   ├── api/                 # 接口层：只负责路由、参数校验、结果返回
│   │   ├── __init__.py
│   │   ├── model.py         # 模型连通性相关接口
│   │   ├── knowledge.py     # 知识库管理接口（上传、构建、列表）
│   │   └── chat.py          # 问答接口
│   ├── core/                # 核心业务层：RAG 核心能力
│   │   ├── __init__.py
│   │   ├── rag_engine.py    # RAG 问答编排逻辑（原 rag_engine.py）
│   │   └── reranker.py      # 语义重排序模块（原 reranker.py）
│   ├── services/            # 服务层：业务逻辑封装，衔接接口与数据层
│   │   ├── __init__.py
│   │   ├── document_service.py  # 文档处理服务
│   │   └── vector_service.py    # 向量库构建与检索服务
│   ├── utils/               # 工具层：通用无状态工具
│   │   ├── __init__.py
│   │   └── doc_processor.py     # 文档解析、清洗、语义分块（原 doc_processor.py）
│   └── db/                  # 数据层：数据持久化交互
│       ├── __init__.py
│       └── vector_db.py         # 向量数据库封装（原 vector_db.py）
├── frontend/                # 前端层：与后端完全解耦
│   └── streamlit_app.py     # 原 frontend.py
├── data/                    # 运行时数据目录（与代码完全隔离）
│   └── knowledge_bases/     # 原 knowledge_bases 目录迁移到此处
├── requirements.txt         # 项目依赖清单
└── README.md                # 项目说明文档
```


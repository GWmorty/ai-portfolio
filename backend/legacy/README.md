# legacy — 已废弃的历史版本，仅作学习参考，线上不再使用

- `server.py`：初版，纯 RAG（检索 + 拼 prompt + DeepSeek 非流式）
- `server_agent.py`：过渡版（LangChain `create_agent`）

当前生产入口：`backend/server_langgraph.py`（LangGraph 白盒 Agent + SSE 流式 + Memory）

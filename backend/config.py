# config.py — 全局共享配置（常量单一来源，避免多文件硬编码漂移）
#
# 之前 CHROMA_PATH / COLLECTION_NAME / base_url / 模型名 在 mini_rag.py、
# server_langgraph.py、eval_rag.py、eval_generation.py 各自硬编码，
# 改一处容易漏改其他文件。现在统一在这里，其他模块 import 使用。

import os

# ========== Chroma（RAG 持久化）==========
# ⚠️ 必须在项目目录外（~/.ai_portfolio/chroma_db）：
# 否则 Next.js dev server 会监控到 sqlite3 文件变化，触发无限刷新
CHROMA_PATH = os.path.expanduser("~/.ai_portfolio/chroma_db")
COLLECTION_NAME = "knowledge_base"

# ========== 模型 ==========
EMBED_MODEL = "BAAI/bge-m3"              # 硅基流动 embedding（1024 维）
RERANK_MODEL = "BAAI/bge-reranker-v2-m3" # 两阶段检索重排（实验用，线上未启用）
CHAT_MODEL = "deepseek-chat"             # DeepSeek 对话模型

# ========== API 端点 ==========
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ========== GitHub 工具 ==========
GITHUB_USERNAME = "GWmorty"
GITHUB_CACHE_TTL = 600  # 结果缓存秒数（未认证 API 限流 60 次/小时/IP，出口 IP 全体访客共用）

# ========== 检索 ==========
TOP_N = 3  # search_candidate_info 默认召回数（线上 eval 按 top3 命中率算）

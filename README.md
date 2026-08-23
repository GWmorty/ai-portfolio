# AI 求职作品集（ai-portfolio）

> 一个让 HR 通过对话了解候选人的求职主页：访客既能看简介、技能、项目经历，
> 也能直接向 AI 助手提问——AI 基于我的真实资料流式回答，并展示检索过程，不编造。
> **这个产品本身，就是我技术能力的演示。**

**线上地址**：https://szds.site （备用 IP 直连：http://43.156.80.35）

## 产品形态

- **求职主页**：Hero / 关于我 / 技能栈 / 工作经历 / 项目展示 / 开发踩坑实录 / 联系方式
- **AI 求职助手**（核心）：聊天界面，访客问"他学过什么技术栈""做过哪些项目""base 在哪里"，
  AI 基于知识库（`backend/data/*.md`）检索后流式回答，思考过程（工具调用步骤）可视化

## 架构

```
浏览器
  │  Next.js 16 静态导出（out/ 由 Nginx 托管）
  │  ChatSection：SSE 逐字渲染 + 匀速缓冲 + 思考过程卡片
  ▼
Nginx（HTTPS: Let's Encrypt · /api/ 反代 · 限流 20r/m）
  ▼
FastAPI（Docker 容器，只绑 127.0.0.1:8000）
  │  LangGraph 白盒 Agent：State(agent↔tools 循环) + MemorySaver 多轮记忆
  │  工具×3：候选人资料检索 / GitHub 信息 / GitHub 仓库（TTL 缓存 + 限流兜底）
  │  POST /ask/stream：SSE 流式（双 stream_mode 解析 token + 工具事件，过滤工具前奏）
  ▼
RAG：markdown H2 语义切块 → BGE-M3 embedding（硅基流动，1024 维）
     → Chroma 余弦检索 top3 → DeepSeek 生成（系统提示词含防编造规则）
```

## 质量与评估（数字说话）

改动检索/切块前必须跑 eval 对比，本仓库的所有技术决策都有实验数据支撑：

| 指标 | 结果 | 说明 |
|------|------|------|
| 检索 Hit@1 / Hit@3 / MRR | 88.2% / 100% / 0.936 | `eval_rag.py`，golden set 34 个 HR 真实问题，文件级判定；知识库 55 chunks |
| 生成忠实度 | 12/12 | `eval_generation.py`，LLM-as-judge 逐句核对事实是否被资料支持；同时报告回答率，防"全拒答刷满分" |
| 负实验记录 | reranker ×2、HyDE、query rewrite | 均未达上线阈值，未上线——小而规整的语料上，bi-encoder 已接近最优 |

关键结论：**切块策略比模型更重要**（字符切块 → H2 语义切块，Hit@1 +6pp）；
**更先进的组件不一定更好**（reranker 两次实验均为负优化，靠 eval 数据而非直觉判断）。

## 技术栈

- **前端**：Next.js 16（App Router，静态导出）+ React 19 + Tailwind CSS 4
- **后端**：Python 3.14 + FastAPI + LangGraph（白盒 Agent）+ LangChain
- **RAG**：BGE-M3（硅基流动）+ Chroma（Docker named volume 持久化）+ DeepSeek
- **部署**：Docker + docker-compose + Nginx + GitHub Actions CI/CD
  （push 到 main 自动部署，部署前 pytest 门禁，失败自动回滚）

## 本地开发

```bash
# 后端（backend/.venv 已按 requirements.txt 全量 == 锁定装好，勿用全局 python）
cd backend
.venv/Scripts/uvicorn server_langgraph:app --port 8000

# 前端（开发模式自动走 localhost:8000）
npm run dev

# 跑评估
cd backend && .venv/Scripts/python eval_rag.py            # 检索质量
cd backend && .venv/Scripts/python eval_generation.py     # 生成忠实度

# 跑测试（CI 门禁同款，共 18 个）
cd backend && .venv/Scripts/python -m pytest -q
```

## 访客观测（轻量自研）

每次提问写入 SQLite（`stats.db`，与 Chroma 同 Docker 卷持久化）：时间、问题、会话、
命中的工具、延迟、最终回答、状态（ok / aborted 中断 / error）。

- **隐私边界**：公开接口 `/api/stats` 只返回聚合计数（首页展示"已有 N 位访客提过
  M 个问题"），不暴露访客问题原文；`eval-%` / `smoke-%` 前缀会话不计入。
- **看访客在问什么**（服务器上，反向优化资料的依据）：

```bash
docker exec ai-portfolio-backend python -c "import sqlite3; [print(r) for r in sqlite3.connect('/root/.ai_portfolio/stats.db').execute(\"SELECT substr(ts,1,10), session_id, latency_ms, tools, question FROM ask_log ORDER BY id DESC LIMIT 30\")]"
```

## 部署须知（三个非显然的坑）

1. **前端改动必须 `npm run build`**：`out/` 在 .gitignore 里，git pull 不会更新线上前端（CI 已自动处理）。
2. **改 `backend/data/*.md` 后必须手动重入库**：Chroma 入库是独立步骤，CI 不包含——
   需清库重建并重启后端（`_ensure_embeddings` 只在空库时算 embedding，直接跑是 no-op）。
3. **依赖全部 `==` 锁定，本地与容器统一 Python 3.14**：曾因 `>=` 导致流式行为漂移；
   升级任何依赖先本地验证再改锁定版本。

## 仓库结构

```
├── app/                  # Next.js 页面（8 个区块组件在 components/）
├── backend/
│   ├── server_langgraph.py   # 生产入口：FastAPI + LangGraph Agent
│   ├── mini_rag.py           # RAG 引擎：H2 语义切块 + embedding + Chroma 入库
│   ├── config.py             # 常量单一来源（路径/模型/端点/检索参数）
│   ├── eval_rag.py           # 检索质量评估（--verbose/--rebuild/--rerank/--rewrite/--hyde）
│   ├── eval_generation.py    # 生成忠实度评估（LLM-as-judge，--production 直测线上）
│   ├── test_*.py             # pytest（切块/配置/路由，CI 门禁）
│   └── data/*.md             # 知识库语料（简历/项目/FAQ，RAG 的数据源）
├── docker-compose.yml    # 后端容器（127.0.0.1:8000）+ chroma_data 卷
└── .github/workflows/deploy.yml   # CI/CD：SSH 部署 + pytest 门禁 + 失败回滚
```

## 关于我

范睿峰 · 上海 · 信息管理与信息系统（学士）· 方向：懂技术的项目管理 / AI 应用层岗位
邮箱：2413824669@qq.com · GitHub：https://github.com/GWmorty

有想问的？直接问线上那个 AI 助手——它比我回得快。

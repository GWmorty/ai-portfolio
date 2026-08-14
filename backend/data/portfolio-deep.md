# AI 求职作品集·工程深挖

> 面试深挖版：这个网站怎么做的、为什么这么做、怎么验证做得对。项目概览见「项目经历」。

## 整体架构与选型
前后端分离：前端 Next.js 16 静态导出（out/ 由服务器 Nginx 托管），后端 FastAPI + LangGraph Agent 跑在 Docker 容器里（只绑 127.0.0.1，外网走 Nginx 反代 /api/）。向量库 Chroma 用 named volume 持久化，容器重建不丢数据。LLM 用 DeepSeek（SSE 流式），embedding 用硅基流动 BGE-M3（1024 维）。选型原则：单机 docker-compose 对求职站流量是够用且诚实的架构，不上 K8s、不过度设计。

## Agent 设计（LangGraph 白盒图）
不用黑盒 Agent 框架，自己搭 State/Node/Edge：agent 节点调 LLM 决定是否用工具，条件边路由到 tools 节点，执行完回到 agent 循环，直到产出最终答案。共 3 个工具：候选人资料检索、GitHub 信息、GitHub 仓库（后两个带 600 秒结果缓存和 403 兜底，因为未认证 GitHub API 限流 60 次/小时且服务器出口 IP 由全体访客共享）。多轮记忆用 MemorySaver checkpointer + 每访客独立 session_id；关闭并行工具调用，强制 ReAct 串行多步推理——复合问题一次查一个方面，查完看结果够不够再决定下一步。

## 流式输出与思考过程可视化
/ask/stream 用 LangGraph 双 stream_mode（messages + updates）同时拿 token 流和节点状态：messages 流出 AIMessageChunk 逐 token 推给前端，updates 流解析出 tool_start / tool_end 事件，前端渲染成「正在检索候选人资料」这类思考过程卡片。踩过的坑：LLM 调工具前常冒英文前奏（Let me search...），解法是缓冲机制——见到 ToolMessage 就丢弃之前的缓冲内容，流结束仍没见到工具才确认是真答案并 flush。

## 检索与评估体系
切块按 markdown H2 标题整段切，每段带文件 H1 标题做上下文（比字符硬切 Hit@1 提升 6 个百分点）。检索质量用自建 eval 框架量化：golden set 是 HR 真实问题 + 期望命中的源文件，评 Hit@1/3/5 和 MRR，评估器与线上检索器完全同款。铁律：改检索必跑 eval 对比，数字涨了才上线。做过三个失败的实验（均为负优化，未上线）——BGE-Reranker 两阶段检索（两次实验都更差）、HyDE 假想文档（明显更差，LLM 编造的通用面试话术在向量空间漂离真实语料）、query rewrite（边际改善，不值一次额外 LLM 调用延迟）。

## 生成忠实度与防编造
LLM-as-judge 逐句核对回答里的每个事实陈述是否有资料依据（12 个 HR 问题全部通过，同时报告回答率，防止「全拒答也能刷满忠实度」的指标漏洞）。系统提示词 9 条规则，核心是：资料里没有的就答「暂时没有详细说明」，禁止自行补充未来计划、数字、技术选型。GitHub 实时数据单独处理：评估裁判会拉取实时 GitHub 数据比对，避免把真实的仓库数误判为编造。

## 部署与 CI/CD
push 到 main 触发 GitHub Actions SSH 到云服务器自动部署：构建后端镜像 → pytest 门禁（测试跑在隔离的临时 Chroma 卷上，不碰生产库）→ 门禁失败自动 git reset 回滚 → docker-compose 重建容器 → 前端 npm run build 重新生成 out/。线上加固：Nginx 接口限流（每分钟 20 次 + burst）、/health/ai 真实 AI 链路探测（max_tokens=1 的最小调用，防「存活绿灯但 key 已失效」的假绿灯）、依赖全量 == 锁定且本地与容器统一 Python 3.14（曾因 >= 静默升级导致流式从逐 token 退化成整条输出）。

## 关键踩坑与取舍
版本漂移（>= 静默升级坑）、Python 3.14 运行时才能逐 token 流式、前端 out/ 不进 git 要单独 build、Chroma 改语料要清库重入库（入库逻辑只在空库时算 embedding）、长驻 uvicorn 进程持有旧库视图所以入库后必须重启、Windows curl 对部分 TLS 链路假失败要用 Python 验证。取舍：MemorySaver 是进程内的，容器重启会清空访客对话记忆——求职站访客会话是短时的，单 worker 部署下可接受，将来多 worker 再换 SQLite checkpointer。

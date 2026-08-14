# 项目经历

## 浦东某中学智慧校园系统（屹力教育科技）
**时间**：2024.11 - 2025.01
**角色**：项目专员

- 协助项目经理整理并汇总用户需求，撰写《需求规格说明书》，确保开发需求与校方实际使用场景匹配
- 协调开发同事跟进任务进度，维护甘特图，定期更新项目状态，帮助团队提前识别潜在延误
- 协助组织用户培训与资料准备，制作基础操作手册与演示视频，降低用户上线初期的咨询量

## 公司标品迭代升级（屹力教育科技）
**时间**：2024.10 - 2025.01
**角色**：项目专员

- 收集 30+ 所学校的使用反馈和改进建议，帮助产品同事整理优先级清单
- 参与制作与测试新版功能，记录并跟踪问题修复情况
- 整理产品文档、操作手册与内部培训资料，支持跨部门培训与知识共享

## AI 求职作品集（个人独立项目）
**时间**：2026.06 - 至今
**技术栈**：Next.js + Tailwind CSS + FastAPI + LangGraph + RAG

### 项目说明
这是为求职打造的个人作品集网站，让 HR 通过与 AI 助手对话来了解候选人。访客体验产品的过程，就是了解我的过程。这是我 AI 工程能力的综合落地项目。

### 已完成
- **前端**：Next.js + Tailwind CSS 实现 5 栏式作品集（Hero、About、Skills、Projects、Contact），PC + 移动端响应式
- **后端**：FastAPI 服务，把 RAG 引擎和 LangGraph Agent 包成 API
- **AI 助手（Agent）**：基于 LangGraph 的白盒 Agent，带 3 个工具（候选人资料检索 / GitHub 信息 / GitHub 仓库），支持多轮记忆（Memory）和 ReAct 多步推理
- **RAG 系统**：BGE-M3 embedding（1024 维）+ Chroma 向量数据库 + 余弦相似度检索；知识库 6 个源文件 + HR 高频问答共 55 个 chunk，按 markdown H2 标题语义切块
- **检索质量评估**：自建 eval 框架，用 34 个 HR 真实问题做 golden set，量化 Hit@1/3/5 + MRR。当前基线：Hit@1=88.2%、Hit@3=100%、Hit@5=100%、MRR=0.936。测过 BGE-Reranker，数据证明在小数据集上反而更差，果断不上线——用数据做技术选型
- **生成忠实度评估**：LLM-as-judge 双模式——纯 RAG 链路与生产 Agent 的 /ask 回答各用 12 个 HR 问题检验，逐句核对每个事实陈述是否有资料依据，目前 12/12 全过（样本持续扩充中，不称"100%"）；同时报告回答率（防"全拒答也能刷满忠实度"的指标漏洞）；系统提示词含防编造规则（资料里没有的未来计划/数字禁止补充）
- **流式体验**：SSE 逐 token 输出 + agent 思考过程可视化（HR 能看到"正在检索候选人资料"这类内部动作）+ 前端匀速缓冲（解决后端攒批导致的卡顿）
- **部署与运维**：Docker 化后端，GitHub Actions CI/CD（push 到 main 自动部署），Nginx 反向代理，HTTPS（Let's Encrypt 自动续期），依赖版本全 == 锁定（本地与容器统一 Python 3.14，根除版本漂移）

### 网址
- 正式域名：https://szds.site（Let's Encrypt HTTPS 自动续期，国内可直接访问）
- 备用 IP 直连：http://43.156.80.35

### GitHub
https://github.com/GWmorty

### 我从项目里学到的
- LangGraph 白盒 Agent 的 State/Node/Edge 模型，以及 stream_mode（messages/updates）做 agent 可观测性
- RAG 不只是"接个向量库"：切块策略比模型更重要，检索质量必须用 eval 量化而不是凭感觉
- "先进的技术 ≠ 更好"：reranker 在我的场景反直觉地更差，靠实验和数据才能判断
- 工程化：版本锁定、CI/CD、可观测性这些"无聊"的事，是项目能稳定运行的关键

## mini RAG 知识库（个人独立项目）
**时间**：2026.07
**技术栈**：Python + BGE-M3 + Chroma + DeepSeek API

### 项目说明
从零实现的 RAG（检索增强生成）知识库，是这个求职作品集 AI 助手的核心引擎。

### 技术亮点
- **切块**：markdown H2 标题语义切块（早期是字符切块，实测 Hit@1 从 81% 升到 87%，用数据驱动升级）
- **embedding**：使用 BGE-M3 模型（1024 维，多语言）
- **向量数据库**：Chroma 持久化，避免重复调用 API，启动时间从秒级降到毫秒级
- **检索**：余弦相似度语义检索（取代 Jaccard 字面匹配）
- **生成**：DeepSeek API + system prompt 约束（防止 AI 编造）
- **引用源展示**：每个回答附带具体来源段落，可追溯

### 升级路径
- Level 1：embedding + Chroma（已完成）
- Level 2：FastAPI 服务化（已完成）
- Level 3：Docker 化部署 + CI/CD（已完成）
- Level 4：Rerank、Hybrid Search 等高级技巧（已实验 BGE-Reranker，数据证明在小数据集上反而更差，未上线）

## zero-to-tech 课程项目
**时间**：2026.05 - 2026.06
**技术栈**：Next.js + Tailwind CSS

跟随李勃老师「零到全栈」课程完成的前端项目，系统学习了：
- HTML/CSS/JavaScript 基础
- React + Next.js 框架
- Tailwind CSS 实用类
- 静态导出 + Nginx 自部署

已完成前端模块（1.1-4.6），正在学后端模块（FastAPI）。

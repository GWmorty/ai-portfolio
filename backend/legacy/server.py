# server.py - 把 miniRGA 包成 FastAPI 服务
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from miniRGA import RAGBot  # 复用我们写的 RAG 引擎


# ========== 启动时初始化 RAG 引擎 ==========
print("🚀 启动中：初始化 RAG 引擎...")
bot = RAGBot("./data")  # 相对路径，指向 backend/data/
print("✅ RAG 引擎就绪\n")


# ========== 创建 FastAPI 应用 ==========
app = FastAPI(
    title="范睿峰 AI 求职助手",
    description="HR 提问，AI 基于候选人资料回答",
    version="0.1.0",
)


# 配置 CORS（允许前端跨域调用，开发期允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改成具体域名，例如 https://ruifeng.me
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ========== 定义请求/响应格式（Pydantic 模型） ==========
class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    source: str
    chunk_index: int
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


# ========== 路由：健康检查 ==========
@app.get("/health")
def health():
    """用于监控：服务器活着吗"""
    return {"status": "ok", "chunks_loaded": bot.collection.count()}


# ========== 路由：核心 /ask 端点 ==========
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """
    HR 提问端点
    请求: {"question": "范睿峰会什么编程语言？"}
    响应: {"answer": "...", "sources": [...]}
    """
    # 1. 检索相关 chunks
    results = bot.retrieve(request.question, top_n=3)

    # 2. 拼接 context
    context = "\n\n".join([chunk["text"] for chunk in results])

    # 3. 构造 system prompt（关键：约束 AI 不编造）
    prompt = f"""你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 只基于下面【参考资料】回答，不要凭空编造
2. 如果资料里没有相关信息，直接说"这方面资料我暂时没有详细说明，欢迎直接联系范睿峰"
3. 不要编造经历、技能或项目
4. 语气专业、简洁、自信
5. 用第一人称回答（"我"指代范睿峰）

【参考资料】
{context}

【HR 的问题】
{request.question}
"""

    # 4. 调 DeepSeek（API 模式不用流式，等完整回答再返回 JSON）
    response = bot.chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    # 5. 返回结构化 JSON（answer + sources）
    return AskResponse(
        answer=answer,
        sources=[
            SourceItem(
                source=r["source"],
                chunk_index=r["chunk_index"],
                preview=r["text"][:100].replace("\n", " "),
            )
            for r in results
        ],
    )

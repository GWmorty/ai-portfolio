# server_langgraph.py
# LangGraph 版 FastAPI 服务：白盒 Agent（完全可控）+ 流式输出
# 启动：uvicorn server_langgraph:app --port 8000

import os
import time
import json
import sqlite3
import threading
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    CHAT_MODEL,
    SILICONFLOW_BASE_URL,
    DEEPSEEK_BASE_URL,
    GITHUB_USERNAME,
    GITHUB_CACHE_TTL,
    TOP_N,
    STATS_DB_PATH,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver  # ⭐ Memory 模块

# RAG 相关
import chromadb
from openai import OpenAI as OpenAIClient

load_dotenv()

# ========== RAG 基础设施（路径/集合名/端点统一在 config.py）==========
embed_client = OpenAIClient(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=SILICONFLOW_BASE_URL,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# ⭐ 健康检查专用：探测 DeepSeek 链路（不走业务 llm，避免和流式配置耦合）
health_chat_client = OpenAIClient(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL,
)


def get_embedding(text):
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


# ========== LLM ==========
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL,
    temperature=0,
    streaming=True,  # ⭐ 开启 token 级别流式（逐字输出）
)


# ========== 工具（RAG + GitHub，三个工具；GitHub 用户名在 config.py）==========


@tool
def search_candidate_info(query: str) -> str:
    """搜索范睿峰的个人信息、技能、项目经历、求职方向、联系方式等。"""
    try:
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_N,
            include=["documents"],
        )
        docs = results["documents"][0] if results["documents"] else []
    except Exception as e:
        # embedding 服务故障（2026-08-23 事故：硅基流动余额耗尽返回 402）：
        # 返回可读错误而不是抛异常——抛异常会让整条流式回答崩掉（访客看到空白），
        # 降级成错误文本让 Agent 能礼貌引导访客直接联系，且不会编造资料。
        return (
            f"资料检索服务暂时不可用（embedding 服务异常：{type(e).__name__}）。"
            "请如实告诉访客：资料检索暂时故障，无法回答该问题，"
            "可直接联系范睿峰：电话 18021080437 / 邮箱 2413824669@qq.com。"
            "严禁在拿不到资料的情况下编造答案。"
        )
    return "\n\n".join(docs) if docs else "没有找到相关信息"


# ⭐ GitHub 结果缓存（TTL 600s）：
# 未认证的 GitHub API 限流很紧（60 次/小时/IP），服务器出口 IP 是所有访客共用，
# 连续被问几次 GitHub 就会 403，必须缓存 + 失败兜底
_GITHUB_CACHE = {}  # key -> (expire_ts, data)


def _github_cached(key: str, fetcher):
    now = time.time()
    hit = _GITHUB_CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        data = fetcher()
    except requests.RequestException:
        # 网络失败/超时：有旧缓存用旧的，没有就返回友好错误（Agent 能据此回答）
        old = _GITHUB_CACHE.get(key)
        if old:
            return old[1]
        return {"error": "GitHub 查询失败（网络或超时），请稍后再试"}
    _GITHUB_CACHE[key] = (now + GITHUB_CACHE_TTL, data)
    return data


@tool
def get_github_info(username: str) -> dict:
    """查询 GitHub 用户的基本信息（仓库数、粉丝数）。"""
    def fetch():
        response = requests.get(
            f"https://api.github.com/users/{username}", timeout=6
        )
        if response.status_code == 403:
            return {"error": "GitHub API 限流中，请稍后再试"}
        if response.status_code != 200:
            return {"error": f"GitHub 用户 {username} 不存在"}
        data = response.json()
        return {
            "username": data["login"],
            "public_repos": data["public_repos"],
            "followers": data["followers"],
        }

    return _github_cached(username, fetch)


@tool
def get_github_repos(username: str) -> dict:
    """查询 GitHub 用户最近的仓库列表。"""
    def fetch():
        response = requests.get(
            f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5",
            timeout=6,
        )
        if response.status_code == 403:
            return {"error": "GitHub API 限流中，请稍后再试"}
        if response.status_code != 200:
            return {"error": "GitHub 仓库列表获取失败"}
        repos = response.json()
        return {
            "repos": [
                {"name": r["name"], "description": r.get("description", "无"), "language": r.get("language", "未知")}
                for r in repos
            ]
        }

    return _github_cached(f"{username}:repos", fetch)


tools = [search_candidate_info, get_github_info, get_github_repos]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)  # 强制串行工具调用，实现 ReAct 多步推理


# ========== System Prompt ==========
SYSTEM_PROMPT = f"""你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 关于范睿峰的个人问题（技能、经历、项目、求职方向）→ 使用 search_candidate_info 工具
2. 关于 GitHub 的问题 → 使用 get_github_info 或 get_github_repos（用户名 {GITHUB_USERNAME}）
3. 不要编造经历或技能
4. 资料里没有的信息，说"暂时没有详细说明"
5. 用第一人称回答（"我"指代范睿峰）
6. 语气专业、简洁、自信
7. **始终用中文回答**，不要用英文
8. **调用工具时不要输出任何文字说明**（不要说"让我查一下""Let me search"之类），直接调用工具，拿到结果后再组织回答
9. **禁止自行补充资料里没有的内容**：未来计划、细节、数字、技术选型等，资料里没有明确写的就不要说；不确定就说"这方面资料我暂时没有详细说明"，不要为了显得完整而编造

【推理方式（ReAct）——遇到复杂问题分步查】
- 复合问题（同时问到多个方面，如"你的技能 + 怎么用到工作上"）→ 拆成多个子问题，
  一次查一个方面，看到结果后再决定是否需要再查下一个，不要一次并行查全部。
- 简单问题（单一方面）→ 查一次即可，不要过度拆解。
- 每次调工具前，心里想清楚"这步要查什么、为什么"；查完看结果够不够再决定下一步。
- 全部所需信息收集齐后，再综合成完整回答，不要在中间就急着答。
"""


# ========== LangGraph 核心：State + Node + Edge ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: AgentState):
    """节点 1：调 LLM（决定是否用工具）"""
    # 每次调用都加 SystemMessage（不存入 state，只用于 LLM 调用）
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)  # ⭐ async + ainvoke
    return {"messages": [response]}


def should_continue(state: AgentState):
    """条件分支：LLM 是否要求调工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# 组装图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")  # 工具执行完回到 agent（循环）

print("初始化 LangGraph Agent（含 Memory）...")
# ⭐ Memory：进程内 MemorySaver（checkpointer）。
# 取舍说明：容器重启 / CI 重建后访客对话记忆会清空——求职站的访客会话是短时的，
# 单 worker 部署下可接受；将来若要多 worker 或需要跨重启持久记忆，
# 再换 SQLite checkpointer（文件放 chroma 同一个卷里）。
agent_graph = workflow.compile(checkpointer=MemorySaver()) # ⭐ 加 checkpointer
print("LangGraph Agent 就绪（支持多轮对话）！\n")


# ========== 访客观测（轻量自研）：每次提问写 SQLite，与 Chroma 同卷持久化 ==========
# 隐私边界：公开 /stats 只返回聚合计数，访客问题原文不对外；
#           想看访客在问什么，SSH 到服务器查库（命令见 README「访客观测」）。
# 口径：eval-%（评估脚本）/ smoke-%（冒烟测试）前缀的会话不计入统计。
_stats_lock = threading.Lock()

_ASK_LOG_DDL = """CREATE TABLE IF NOT EXISTS ask_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    answer TEXT,
    tools TEXT
)"""


def _log_ask(session_id, question, status, latency_ms, answer="", tools=""):
    with _stats_lock:
        conn = sqlite3.connect(STATS_DB_PATH, timeout=5)
        try:
            conn.execute(_ASK_LOG_DDL)
            conn.execute(
                "INSERT INTO ask_log (ts, session_id, question, status, latency_ms, answer, tools)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    session_id, question, status, latency_ms, answer, tools,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def _stats_counts():
    """聚合计数（公开）：成功回答的问题数 + 去重访客数（按 session_id 近似）"""
    with _stats_lock:
        conn = sqlite3.connect(STATS_DB_PATH, timeout=5)
        try:
            conn.execute(_ASK_LOG_DDL)
            questions, visitors = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT session_id) FROM ask_log"
                " WHERE status='ok' AND session_id NOT LIKE 'eval-%' AND session_id NOT LIKE 'smoke-%'"
            ).fetchone()
            return {"questions": questions, "visitors": visitors}
        finally:
            conn.close()


# ========== FastAPI 应用 ==========
app = FastAPI(
    title="范睿峰 AI 求职助手（LangGraph 版）",
    description="LangGraph 白盒 Agent + RAG + GitHub API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # 正式域名 + 本地开发（npm run dev 是 localhost:3000，预检否则被拒）
    allow_origins=["https://szds.site", "https://www.szds.site", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    # 长度上限：nginx 限流只挡频率，挡不住单次请求体积；超长输入会直接烧 LLM token（校验失败 FastAPI 返回 422）
    question: str = Field(..., min_length=1, max_length=500)
    session_id: str = Field(default="default", max_length=100)  # ⭐ 标识对话（相同 = 同一对话，有记忆）


class AskResponse(BaseModel):
    answer: str
    agent_used: bool = True


@app.get("/health")
def health():
    """存活探针：进程在不在 + 基础资源状态（不做外部调用，保持轻快）"""
    return {
        "status": "ok",
        "agent": "LangGraph + Memory",
        "model": CHAT_MODEL,
        "chunks": collection.count(),
        "keys": {
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "siliconflow": bool(os.getenv("SILICONFLOW_API_KEY")),
        },
    }


@app.get("/health/ai")
async def health_ai():
    """真实 AI 链路探测：DeepSeek（对话）+ 硅基流动（embedding）各做一次最小调用。
    防「假绿灯」——key 配置存在 ≠ 可用：2026-08-14 DeepSeek key 失效 401、
    2026-08-23 硅基流动余额耗尽 402，两次事故里 /health 都依然全绿。
    任一 Provider 异常即返回 503（docker healthcheck 探测本端点）。"""
    import asyncio
    results = {}
    try:
        await asyncio.to_thread(
            health_chat_client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=10,
        )
        results["deepseek"] = "ok"
    except Exception as e:
        results["deepseek"] = type(e).__name__
    try:
        await asyncio.to_thread(
            embed_client.embeddings.create,
            model=EMBED_MODEL,
            input=["ping"],
            timeout=10,
        )
        results["siliconflow"] = "ok"
    except Exception as e:
        results["siliconflow"] = type(e).__name__
    if any(v != "ok" for v in results.values()):
        raise HTTPException(status_code=503, detail=f"AI 链路异常: {results}")
    return {"status": "ok", "ai": results, "model": CHAT_MODEL}


@app.get("/stats")
def stats_endpoint():
    """公开聚合计数（首页社会证明用）。只返回数量，不含访客问题原文。"""
    return _stats_counts()


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    # 节点 call_model 是 async def，必须用 ainvoke；用同步 invoke 会在
    # LangGraph 1.x 报 TypeError: No synchronous function provided to "agent"
    t0 = time.time()
    try:
        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content=request.question)]},
            config={"configurable": {"thread_id": request.session_id}},
        )
        answer = result["messages"][-1].content
        tools_used = ",".join(sorted({
            m.name for m in result["messages"] if isinstance(m, ToolMessage)
        }))
    except Exception:
        _log_ask(request.session_id, request.question, "error", int((time.time() - t0) * 1000))
        raise
    _log_ask(request.session_id, request.question, "ok", int((time.time() - t0) * 1000), answer, tools_used)
    return AskResponse(answer=answer, agent_used=True)


# ⭐ 流式端点：逐字返回 AI 回答（像 ChatGPT 打字机效果）
@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """SSE 流式输出——逐字返回 AI 回答 + agent 工具调用过程"""
    async def generate():
        # 工具名 → 中文友好标签（前端可读）
        TOOL_LABEL = {
            "search_candidate_info": "检索候选人资料",
            "get_github_info": "查 GitHub 资料",
            "get_github_repos": "查 GitHub 仓库",
        }
        seen_tool_msg = False  # 是否已出现过 ToolMessage（出现后，后续 AIMessageChunk 才是真答案）
        pending = []           # 缓冲"尚未确认是真答案"的 AIMessageChunk 文本（可能是工具前奏）
        # 访客观测：工具集合 / 答案累计 / 状态（正常结束 ok、访客中断 aborted、异常 error）
        t0 = time.time()
        tools_used, answer_parts, status = set(), [], "ok"
        try:
            async for chunk in agent_graph.astream(
                {"messages": [HumanMessage(content=request.question)]},
                config={"configurable": {"thread_id": request.session_id}},
                stream_mode=["messages", "updates"],
            ):
                mode, payload = chunk  # (stream_mode_name, data)

                if mode == "messages":
                    msg = payload[0]   # (message, metadata) 里的 message
                    is_ai = isinstance(msg, (AIMessageChunk, AIMessage))
                    is_tool = isinstance(msg, ToolMessage)
                    has_text = isinstance(msg.content, str) and bool(msg.content)
                    no_tool_calls = not getattr(msg, "tool_calls", None)

                    if is_tool:
                        # 工具结果到了：说明之前缓冲的 AIMessageChunk 是"工具前奏"（如 Let me search...），丢弃
                        pending = []
                        seen_tool_msg = True
                    elif is_ai and has_text and no_tool_calls:
                        if seen_tool_msg:
                            # 工具之后的 AIMessage = 真正的最终答案，直接透出
                            yield f"data: {json.dumps({'type': 'token', 'content': msg.content}, ensure_ascii=False)}\n\n"
                            answer_parts.append(msg.content)
                        else:
                            # 还没见过工具：可能是「不调工具直接答」（真答案），也可能是「工具前奏」。
                            # 先缓冲，等确认（见 ToolMessage 就丢，流结束没见就 flush）
                            pending.append(msg.content)

                elif mode == "updates":
                    # payload: {node_name: state_delta}。按节点发工具事件
                    for node, state in (payload.items() if isinstance(payload, dict) else []):
                        if not isinstance(state, dict):
                            continue
                        msgs = state.get("messages") or []
                        if not msgs:
                            continue
                        last = msgs[-1]
                        if node == "agent":
                            # agent 节点：LLM 这一步要调工具 → 发 tool_start
                            for tc in getattr(last, "tool_calls", []) or []:
                                tool = tc.get("name", "")
                                args = tc.get("args", {}) or {}
                                query = args.get("query") or args.get("username") or ""
                                tools_used.add(tool)
                                yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool, 'label': TOOL_LABEL.get(tool, tool), 'query': str(query)[:40]}, ensure_ascii=False)}\n\n"
                        elif node == "tools":
                            # tools 节点：ToolMessage 是工具返回值 → 发 tool_end
                            if isinstance(last, ToolMessage):
                                tool = getattr(last, "name", "") or ""
                                out = last.content if isinstance(last.content, str) else str(last.content)
                                yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool, 'label': TOOL_LABEL.get(tool, tool), 'preview': out[:80]}, ensure_ascii=False)}\n\n"
            # 流结束：若 pending 还有内容（说明没调工具，那是直接答的真答案），flush
            if pending:
                answer_parts.append("".join(pending))
                yield f"data: {json.dumps({'type': 'token', 'content': ''.join(pending)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except GeneratorExit:
            # 访客中途关闭页面/发新问题断开连接：也记一笔（"问到一半走了"本身是有价值的信号）
            status = "aborted"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            _log_ask(
                request.session_id, request.question, status,
                int((time.time() - t0) * 1000), "".join(answer_parts),
                ",".join(sorted(tools_used)),
            )

    return StreamingResponse(generate(), media_type="text/event-stream")

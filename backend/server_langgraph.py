# server_langgraph.py
# LangGraph 版 FastAPI 服务：白盒 Agent（完全可控）
# 对比 server_agent.py（create_agent 黑盒）→ 启动：uvicorn server_langgraph:app --port 8000

import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver  # ⭐ Memory 模块

# RAG 相关
import chromadb
from openai import OpenAI as OpenAIClient

load_dotenv()

# ========== RAG 基础设施（和 server_agent.py 相同）==========
CHROMA_PATH = os.path.expanduser("~/.ai_portfolio/chroma_db")
COLLECTION_NAME = "knowledge_base"

embed_client = OpenAIClient(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def get_embedding(text):
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=text)
    return response.data[0].embedding


# ========== LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ========== 工具（RAG + GitHub，三个工具）==========
GITHUB_USERNAME = "GWmorty"


@tool
def search_candidate_info(query: str) -> str:
    """搜索范睿峰的个人信息、技能、项目经历、求职方向、联系方式等。"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    return "\n\n".join(docs) if docs else "没有找到相关信息"


@tool
def get_github_info(username: str) -> dict:
    """查询 GitHub 用户的基本信息（仓库数、粉丝数）。"""
    response = requests.get(f"https://api.github.com/users/{username}")
    if response.status_code != 200:
        return {"error": f"用户 {username} 不存在"}
    data = response.json()
    return {
        "username": data["login"],
        "public_repos": data["public_repos"],
        "followers": data["followers"],
    }


@tool
def get_github_repos(username: str) -> dict:
    """查询 GitHub 用户最近的仓库列表。"""
    response = requests.get(
        f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5"
    )
    repos = response.json()
    return {
        "repos": [
            {"name": r["name"], "description": r.get("description", "无"), "language": r.get("language", "未知")}
            for r in repos
        ]
    }


tools = [search_candidate_info, get_github_info, get_github_repos]
llm_with_tools = llm.bind_tools(tools)


# ========== System Prompt ==========
SYSTEM_PROMPT = f"""你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 关于范睿峰的个人问题（技能、经历、项目、求职方向）→ 使用 search_candidate_info 工具
2. 关于 GitHub 的问题 → 使用 get_github_info 或 get_github_repos（用户名 {GITHUB_USERNAME}）
3. 不要编造经历或技能
4. 资料里没有的信息，说"暂时没有详细说明"
5. 用第一人称回答（"我"指代范睿峰）
6. 语气专业、简洁、自信
"""


# ========== LangGraph 核心：State + Node + Edge ==========

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    """节点 1：调 LLM（决定是否用工具）"""
    # 每次调用都加 SystemMessage（不存入 state，只用于 LLM 调用）
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
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
agent_graph = workflow.compile(checkpointer=MemorySaver()) # ⭐ 加 checkpointer
print("LangGraph Agent 就绪（支持多轮对话）！\n")


# ========== FastAPI 应用 ==========
app = FastAPI(
    title="范睿峰 AI 求职助手（LangGraph 版）",
    description="LangGraph 白盒 Agent + RAG + GitHub API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"  # ⭐ 标识对话（相同 = 同一对话，有记忆）


class AskResponse(BaseModel):
    answer: str
    agent_used: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "agent": "LangGraph + Memory", "model": "deepseek-chat"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    # ⭐ 传 thread_id 让 Agent 记住同一 session 的对话历史
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=request.question)]},
        config={"configurable": {"thread_id": request.session_id}},
    )
    answer = result["messages"][-1].content
    return AskResponse(answer=answer, agent_used=True)

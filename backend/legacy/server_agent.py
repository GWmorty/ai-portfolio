# server_agent.py
# Agent 版 FastAPI 服务：RAG + GitHub 统一为工具
# 对比 server.py（原版纯 RAG）→ 启动命令：uvicorn server_agent:app --port 8000

import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

# RAG 相关（保留原版的 Chroma + embedding）
import chromadb
from openai import OpenAI as OpenAIClient

load_dotenv()

# ========== 初始化 RAG 基础设施 ==========
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
    """调硅基流动 BGE-M3 获取 embedding"""
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=text)
    return response.data[0].embedding


# ========== 创建 LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)


# ========== 定义工具（RAG + GitHub 统一为工具）==========

GITHUB_USERNAME = "GWmorty"


@tool
def search_candidate_info(query: str) -> str:
    """搜索范睿峰的个人信息、技能、项目经历、求职方向、联系方式等。当用户问关于范睿峰本人的任何问题时使用此工具。"""
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
    """查询 GitHub 用户的基本信息（仓库数、粉丝数）。当用户问 GitHub 相关数据时使用。"""
    response = requests.get(f"https://api.github.com/users/{username}")
    if response.status_code != 200:
        return {"error": f"用户 {username} 不存在"}
    data = response.json()
    return {
        "username": data["login"],
        "public_repos": data["public_repos"],
        "followers": data["followers"],
        "html_url": data["html_url"],
    }


@tool
def get_github_repos(username: str) -> dict:
    """查询 GitHub 用户最近的仓库列表（名称、描述、语言）。当用户想了解 GitHub 项目详情时使用。"""
    response = requests.get(
        f"https://api.github.com/users/{username}/repos?sort=updated&per_page=5"
    )
    if response.status_code != 200:
        return {"error": "无法获取仓库"}
    repos = response.json()
    return {
        "repos": [
            {
                "name": r["name"],
                "description": r.get("description", "无描述"),
                "language": r.get("language", "未知"),
            }
            for r in repos
        ]
    }


# ========== 创建 Agent ==========
SYSTEM_PROMPT = f"""你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 关于范睿峰的个人问题（技能、经历、项目、求职方向、联系方式）→ 使用 search_candidate_info 工具
2. 关于 GitHub 的问题 → 使用 get_github_info 或 get_github_repos 工具（用户名是 {GITHUB_USERNAME}）
3. 不要编造经历或技能
4. 资料里没有的信息，说"这方面资料我暂时没有详细说明"
5. 用第一人称回答（"我"指代范睿峰）
6. 语气专业、简洁、自信
7. 范睿峰的 GitHub 用户名是 {GITHUB_USERNAME}，不需要问用户
"""

print("初始化 Agent...")
agent = create_agent(
    model=llm,
    tools=[search_candidate_info, get_github_info, get_github_repos],
)
print("Agent 就绪！\n")


# ========== FastAPI 应用 ==========
app = FastAPI(
    title="范睿峰 AI 求职助手（Agent 版）",
    description="LangChain Agent + RAG + GitHub API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    agent_used: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "agent": "LangChain v1", "model": "deepseek-chat"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """Agent 版 /ask 端点"""
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.question},
            ]
        }
    )
    answer = result["messages"][-1].content
    return AskResponse(answer=answer, agent_used=True)

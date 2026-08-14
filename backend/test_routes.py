# test_routes.py — /ask 与 /ask/stream 路由冒烟测试（CI 门禁用，零外部 API）
#
# 用 FakeGraph 替换 agent_graph：不发真实 LLM/检索请求，专测 FastAPI 端点自身逻辑——
# SSE 事件结构、工具前奏过滤（英文 "Let me search..." 不进 token）、ToolMessage 原文
# 不泄漏进回答、无工具时 pending 缓冲的 flush 分支。
# 这正是部署服务面上历史上出过 bug 的路径（工具前奏泄漏、ToolMessage 泄漏），
# 此前门禁对它零覆盖。

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

import server_langgraph


def _tool_call(name="search_candidate_info", **args):
    return {"name": name, "args": args, "id": "t1", "type": "tool_call"}


# 场景 A：标准工具轮（英文前奏 → tool_calls → ToolMessage → 中文答案）
SCRIPT_WITH_TOOL = [
    ("messages", (AIMessageChunk(content="Let me search "), {})),
    ("messages", (AIMessageChunk(content="his projects."), {})),
    ("updates", {"agent": {"messages": [AIMessage(content="", tool_calls=[_tool_call(query="项目")])]}}),
    ("messages", (ToolMessage(content="RAW-CHUNK-DOCUMENT", name="search_candidate_info", tool_call_id="t1"), {})),
    ("updates", {"tools": {"messages": [ToolMessage(content="RAW-CHUNK-DOCUMENT", name="search_candidate_info", tool_call_id="t1")]}}),
    ("messages", (AIMessageChunk(content="我做过 "), {})),
    ("messages", (AIMessageChunk(content="AI 作品集。"), {})),
]

# 场景 B：不调工具直接答（覆盖 pending 缓冲在流结束时 flush 的分支）
SCRIPT_DIRECT = [
    ("messages", (AIMessageChunk(content="你好，"), {})),
    ("messages", (AIMessageChunk(content="我是范睿峰。"), {})),
]


class FakeGraph:
    def __init__(self, script):
        self._script = script

    async def ainvoke(self, inputs, config=None):
        return {"messages": [HumanMessage("q"), AIMessage("模拟答案")]}

    async def astream(self, inputs, config=None, stream_mode=None):
        for item in self._script:
            yield item


@pytest.fixture
def client_with_tool(monkeypatch):
    monkeypatch.setattr(server_langgraph, "agent_graph", FakeGraph(SCRIPT_WITH_TOOL))
    return TestClient(server_langgraph.app)


@pytest.fixture
def client_direct(monkeypatch):
    monkeypatch.setattr(server_langgraph, "agent_graph", FakeGraph(SCRIPT_DIRECT))
    return TestClient(server_langgraph.app)


def _events(resp_text):
    out = []
    for line in resp_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            out.append(line[5:].strip())
    return out


def test_ask_endpoint(client_with_tool):
    r = client_with_tool.post("/ask", json={"question": "项目", "session_id": "t"})
    assert r.status_code == 200
    assert r.json()["answer"] == "模拟答案"


def test_stream_token_excludes_preamble_and_tool_leak(client_with_tool):
    r = client_with_tool.post("/ask/stream", json={"question": "项目", "session_id": "t"})
    assert r.status_code == 200
    assert r.text.rstrip().endswith("data: [DONE]")

    tokens, kinds = [], []
    for p in _events(r.text):
        if p == "[DONE]":
            continue
        obj = json.loads(p)
        kinds.append(obj.get("type"))
        if obj.get("type") == "token":
            tokens.append(obj.get("content", ""))

    joined = "".join(tokens)
    # 工具前的英文独白不得混进最终回答
    assert "Let me search" not in joined
    # 检索回来的 ToolMessage 原文不得泄漏进 token 流
    assert "RAW-CHUNK-DOCUMENT" not in joined
    assert joined == "我做过 AI 作品集。"
    # 工具过程事件成对出现（思考过程可视化的数据源）
    assert kinds.count("tool_start") == 1
    assert kinds.count("tool_end") == 1


def test_stream_direct_answer_flush(client_direct):
    # 不调工具直接答：token 走 pending 缓冲，流结束时必须 flush（不能丢）
    r = client_direct.post("/ask/stream", json={"question": "你好", "session_id": "t"})
    assert r.status_code == 200
    tokens = []
    for p in _events(r.text):
        if p == "[DONE]":
            continue
        obj = json.loads(p)
        if obj.get("type") == "token":
            tokens.append(obj.get("content", ""))
    assert "".join(tokens) == "你好，我是范睿峰。"

# test_config.py — 配置一致性 + 健康检查探针形状（不调外部 API）
#
# 运行：backend 目录下 `pytest test_config.py -v`

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def test_chroma_path_outside_project():
    # chroma 必须在项目目录外（Next.js dev server 会监控 sqlite 变化导致无限刷新）
    assert ".ai_portfolio" in config.CHROMA_PATH
    assert "chroma_db" in config.CHROMA_PATH


def test_model_and_url_configured():
    assert config.EMBED_MODEL
    assert config.CHAT_MODEL
    assert config.SILICONFLOW_BASE_URL.startswith("https://")
    assert config.DEEPSEEK_BASE_URL.startswith("https://")


def test_health_probe_shape():
    # 直接调 /health 函数（不做 HTTP），验证探针返回结构防回归
    import server_langgraph

    data = server_langgraph.health()
    assert data["status"] == "ok"
    assert isinstance(data["chunks"], int)
    assert isinstance(data["keys"], dict)
    assert "deepseek" in data["keys"] and "siliconflow" in data["keys"]

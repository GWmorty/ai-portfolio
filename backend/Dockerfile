# backend/Dockerfile
# 构建后端 Docker 镜像（Python + FastAPI + LangGraph）

# 基础镜像：Python 3.10 精简版
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 装系统依赖（chromadb 某些依赖需要编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 先装 Python 依赖（利用 Docker 缓存：代码变了不用重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口（FastAPI 默认 8000）
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "server_langgraph:app", "--host", "0.0.0.0", "--port", "8000"]

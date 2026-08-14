import os
import chromadb                     # ⭐ 新增
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    SILICONFLOW_BASE_URL,
    DEEPSEEK_BASE_URL,
)

load_dotenv()

# ⭐ Chroma 路径/集合名统一在 config.py
# 注意：chroma_db 必须存在项目目录外（~/.ai_portfolio/chroma_db）
# 否则 Next.js dev server 会监控到 sqlite3 文件变化，触发无限刷新


# ========== 切块部分 ==========
def chunk_file(file_path, chunk_size=300):
    """
    按 markdown 标题语义切块。
    规则：把文件按 ##（H2）标题切成段落；每段加上文件 H1 标题做上下文。
    没有标题的文件（或 H2 之间的内容）仍按 chunk_size 字符兜底切，避免单块过大。
    返回 [{"text", "source", "chunk_index"}]。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    file_h1 = ""
    sections = []  # [(section_title, body)]
    current_title = ""
    current_body = []

    for line in text.split("\n"):
        if line.startswith("# ") and not file_h1:
            file_h1 = line[2:].strip()
        elif line.startswith("## "):
            # 遇到新 H2：把上一段收尾
            if current_title or current_body:
                sections.append((current_title, "\n".join(current_body)))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    # 最后一段
    if current_title or current_body:
        sections.append((current_title, "\n".join(current_body)))

    chunks = []
    base_name = os.path.basename(file_path)
    for idx, (title, body) in enumerate(sections):
        body = body.strip()
        if not body:
            continue
        # 文件顶部没 H2 的引导段（title 空）直接用，不重复加标题
        context = f"# {file_h1}\n## {title}\n" if title else (f"# {file_h1}\n" if file_h1 else "")
        full = context + body
        # 兜底：单个 section 太长（>4×chunk_size）按字符切
        if len(full) > chunk_size * 4:
            for i in range(0, len(full), chunk_size):
                chunks.append({
                    "text": full[i:i + chunk_size],
                    "source": base_name,
                    "section": title or file_h1 or base_name,
                    "chunk_index": len(chunks),
                })
        else:
            chunks.append({
                "text": full,
                "source": base_name,
                "section": title or file_h1 or base_name,
                "chunk_index": len(chunks),
            })
    return chunks


def chunk_directory(dir_path, chunk_size=300):
    all_chunks = []
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith((".txt", ".md")):
            file_chunks = chunk_file(os.path.join(dir_path, filename), chunk_size)
            for chunk in file_chunks:
                chunk["id"] = len(all_chunks)
                all_chunks.append(chunk)
    return all_chunks


def get_embedding(text, client):
    """调硅基流动 BGE-M3，返回 1024 维向量"""
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


# ========== 修改：RAGBot 类（集成 Chroma）==========
class RAGBot:
    def __init__(self, dir_path, chunk_size=300):
        # 1. 切块
        self.chunks = chunk_directory(dir_path, chunk_size)

        # 2. 两个 client
        self.embed_client = OpenAI(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=SILICONFLOW_BASE_URL
        )
        self.chat_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL
        )

        # ⭐ 3. 新增：Chroma 持久化 client
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # 用余弦距离
        )

        # ⭐ 4. 新增：检查是否需要算 embedding
        self._ensure_embeddings()

    def _ensure_embeddings(self):
        """
        ⭐ 新增：智能加载
        - collection 为空：调 API 算 embedding 入库（首次启动）
        - collection 有数据：直接加载（后续启动）
        """
        if self.collection.count() > 0:
            print(f"✅ 从 Chroma 加载 {self.collection.count()} 个 chunks（无需调 API）")
            return

        print(f"🔧 首次启动：为 {len(self.chunks)} 个文本块生成 embedding...")

        # ⭐ 批量调用 API（比循环调用快得多）
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = []
        batch_size = 32  # BGE-M3 单次最多 32 个
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embed_client.embeddings.create(
                model=EMBED_MODEL,
                input=batch  # 注意这里传的是列表
            )
            # response.data 按 input 顺序返回
            for item in response.data:
                embeddings.append(item.embedding)
            print(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)}")

        # ⭐ 入库（一次 add 全部）
        self.collection.add(
            ids=[str(chunk["id"]) for chunk in self.chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"source": chunk["source"], "section": chunk.get("section", ""),
                 "chunk_index": chunk["chunk_index"]}
                for chunk in self.chunks
            ]
        )
        print(f"✅ Embedding 入库完成，存到 {CHROMA_PATH}\n")

    def retrieve(self, query, top_n=3):
        """⭐ 修改：用 Chroma ANN 检索"""
        # 1. 算 query embedding
        query_embedding = get_embedding(query, self.embed_client)

        # ⭐ 2. Chroma 自动做 ANN 检索（不再手写余弦相似度）
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"]
        )

        # 3. 输出（Chroma 的 distance 是 1 - cosine_similarity）
        print("【检索结果】")
        retrieved = []
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance  # ⭐ 转换：距离→相似度
            preview = doc[:60].replace("\n", " ")
            print(f"  相似度: {similarity:.4f} | 来源: {meta['source']} | 内容: {preview}...")
            retrieved.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta.get("chunk_index", "?")  # ⭐ 新增
            })
        return retrieved

    def ask(self, query, top_n=3):
        results = self.retrieve(query, top_n=top_n)
        print(f"【AI】 ", end="", flush=True)

        context = "\n\n".join([chunk["text"] for chunk in results])
        prompt = f"根据以下参考资料回答问题，资料里没有就说不知道。\n\n【参考资料】\n{context}\n\n【问题】{query}"

        try:
            response = self.chat_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            full_reply = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    full_reply += delta
            print("\n")
            
            # ⭐ 新增：打印引用源
            print("=" * 50)
            print("📚 引用源：")
            for i, chunk in enumerate(results, 1):
                print(f"  [{i}] {chunk['source']} (块 {chunk['chunk_index']})")
                preview = chunk["text"][:80].replace("\n", " ")
                print(f"      {preview}...")
                print()
            
            return full_reply
        except OpenAIError as e:
            print(f"请求失败: {e}")
            return None


if __name__ == "__main__":
    # 命令行测试入口（用于调试，不启动 Web 服务）
    # 正式使用请跑：uvicorn server_langgraph:app --port 8000
    bot = RAGBot("./data")
    bot.ask("范睿峰会什么编程语言？")
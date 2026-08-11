# eval_rag.py
# RAG 检索质量评估（Retrieval Eval）
# 评估对象：和 server_langgraph.py 里 search_candidate_info 完全相同的检索器
#           （BGE-M3 embedding + Chroma 余弦相似度，top-k 召回）
# 指标：
#   Hit@k —— 前 k 条召回里，有没有命中「该问题对应的源文件」（1/0）
#   MRR   —— 第一条命中的倒数排名（1/rank，未命中=0）
# 用法（backend 目录下）：
#   .venv/Scripts/python eval_rag.py            # 汇总
#   .venv/Scripts/python eval_rag.py --verbose  # 额外打印每条召回的 chunk 预览与相似度

import os
import sys
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# Windows 控制台默认 GBK，遇到 ✓/⚠/中文会 UnicodeEncodeError，强制 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ========== 检索器（与 server_langgraph.py 完全一致，评的就是线上那个）==========
CHROMA_PATH = os.path.expanduser("~/.ai_portfolio/chroma_db")
COLLECTION_NAME = "knowledge_base"

embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
)


def get_embedding(text):
    r = embed_client.embeddings.create(model="BAAI/bge-m3", input=text)
    return r.data[0].embedding


def retrieve(query, k=5):
    """返回 [(meta, doc, similarity), ...]，按相似度从高到低"""
    emb = get_embedding(query)
    res = collection.query(
        query_embeddings=[emb], n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for m, d, dist in zip(
        res["metadatas"][0], res["documents"][0], res["distances"][0]
    ):
        out.append((m, d, 1.0 - dist))  # cosine 距离 → 相似度
    return out


# ========== Golden Set：HR 真实问题 + 期望命中的源文件 ==========
# granularity = 文件级（一个文件被切成多个 300 字符 chunk，
#              同一问题的答案可能落在该文件的任意 chunk，所以按文件判命中）
GOLDEN = [
    # ---- skills.md ----
    {"q": "你会哪些编程语言？",              "expect": {"skills.md"}},
    {"q": "Python 你掌握到什么程度？",       "expect": {"skills.md"}},
    {"q": "你会用 Tableau 做什么？",         "expect": {"skills.md"}},
    {"q": "你的 RAG / AI 工程能力怎样？",    "expect": {"skills.md"}},
    # ---- projects.md ----
    {"q": "你做过哪些项目？",                "expect": {"projects.md"}},
    {"q": "mini RAG 知识库用了什么技术？",   "expect": {"projects.md"}},
    {"q": "AI 求职作品集这个项目是什么？",   "expect": {"projects.md"}},
    # ---- work-experience.md ----
    {"q": "你在临腾数字科技做什么？",        "expect": {"work-experience.md"}},
    {"q": "你上一份工作是什么？离职原因？",  "expect": {"work-experience.md"}},
    {"q": "你在上海屹力教育的工作内容？",    "expect": {"work-experience.md", "projects.md"}},
    # ---- job-target.md ----
    {"q": "你的求职方向 / 期望岗位？",       "expect": {"job-target.md"}},
    {"q": "你期望薪资多少？",                "expect": {"job-target.md"}},
    {"q": "你想去哪些行业？",                "expect": {"job-target.md"}},
    # ---- profile.md ----
    {"q": "你的学历和学校？",                "expect": {"profile.md"}},
    {"q": "你的联系方式 / 手机号？",         "expect": {"profile.md"}},
    {"q": "你现在 base 在哪里？",            "expect": {"profile.md"}},
]


def hit_at_k(retrieved, expected, k):
    sources = {m["source"] for m, _, _ in retrieved[:k]}
    return 1 if (sources & expected) else 0


def reciprocal_rank(retrieved, expected):
    for rank, (m, _, _) in enumerate(retrieved, 1):
        if m["source"] in expected:
            return 1.0 / rank
    return 0.0


def main():
    verbose = "--verbose" in sys.argv

    # 自举：本地库为空就先入库（复用 miniRGA 的入库逻辑）
    if collection.count() == 0:
        print(f"Chroma 为空（count=0），先调 BGE-M3 入库 ./data ...")
        from miniRGA import RAGBot
        RAGBot("./data")
    print(f"知识库就绪：{collection.count()} chunks | 评估 {len(GOLDEN)} 个问题\n")

    sums = {"h1": 0, "h3": 0, "h5": 0, "rr": 0.0}
    print(f"{'#':>2} {'H@1':>4} {'H@3':>4} {'H@5':>4} {'RR':>5}  问题")
    print("-" * 60)
    for i, item in enumerate(GOLDEN, 1):
        ret = retrieve(item["q"], k=5)
        h1 = hit_at_k(ret, item["expect"], 1)
        h3 = hit_at_k(ret, item["expect"], 3)
        h5 = hit_at_k(ret, item["expect"], 5)
        rr = reciprocal_rank(ret, item["expect"])
        sums["h1"] += h1
        sums["h3"] += h3
        sums["h5"] += h5
        sums["rr"] += rr

        got3 = "/".join(sorted({m["source"] for m, _, _ in ret[:3]}))
        exp = "/".join(sorted(item["expect"]))
        mark = "✓" if h3 else "✗"
        print(f"{i:>2}{mark}{h1:>4} {h3:>4} {h5:>4} {rr:>5.2f}  {item['q']}")
        print(f"    期望[{exp}]  召回[{got3}]")
        if verbose:
            for m, d, sim in ret[:3]:
                preview = d[:48].replace("\n", " ")
                print(f"      {m['source']}#{m['chunk_index']} sim={sim:.3f} | {preview}")
        if not h3:
            # 失败案例重点标出，方便定位
            print(f"    ⚠ 未命中——可能需要调切块/检索策略")

    n = len(GOLDEN)
    print("\n" + "=" * 40)
    print(f"汇总 (n={n})")
    print(f"  Hit@1 = {sums['h1']/n:.1%}   （top1 就对）")
    print(f"  Hit@3 = {sums['h3']/n:.1%}   （top3 有命中，线上用的是 top3）")
    print(f"  Hit@5 = {sums['h5']/n:.1%}")
    print(f"  MRR   = {sums['rr']/n:.3f}   （越接近 1 越好，1=每次 top1 就命中）")
    print("=" * 40)


if __name__ == "__main__":
    main()

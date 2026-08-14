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
#   .venv/Scripts/python eval_rag.py --rewrite  # 实验：检索前 LLM 改写 query（线上未启用）
#   .venv/Scripts/python eval_rag.py --hyde     # 实验：检索前 LLM 生成假想文档再 embed（线上未启用）

import os
import sys
import requests
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    RERANK_MODEL,
    CHAT_MODEL,
    SILICONFLOW_BASE_URL,
    DEEPSEEK_BASE_URL,
)

# Windows 控制台默认 GBK，遇到 ✓/⚠/中文会 UnicodeEncodeError，强制 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ========== 检索器（与 server_langgraph.py 完全一致，评的就是线上那个；常量在 config.py）==========
embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=SILICONFLOW_BASE_URL,
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
)


def get_embedding(text):
    r = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def rerank(query, documents, top_n=None):
    """调硅基流动 BGE-Reranker（cross-encoder）给文档重排。返回 [(orig_index, score), ...] 降序。"""
    resp = requests.post(
        f"{SILICONFLOW_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "return_top": top_n,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [(item["index"], item["relevance_score"]) for item in resp.json()["results"]]


# 两阶段检索参数
RECALL_K = 10   # bi-encoder 先召回 top-10
RERANK_TOP = 5  # cross-encoder 重排后取前 5


# ========== 查询改写 / HyDE（实验用，线上未启用）==========
# 思路：口语化/中英混用的 query（如「base 在哪」）和语料的语义空间有缝隙，
#       检索前先用 LLM 把 query 变换成更贴近语料风格的文本再 embed。
#   --rewrite：LLM 把问题改写成规范中文检索查询（快，1 次小生成）
#   --hyde   ：LLM 先生成一段"假想资料"（Hypothetical Document），用它检索
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL,
)

REWRITE_PROMPT = (
    "你是检索查询改写器。把下面的 HR 口语化问题（可能中英混杂、指代模糊）"
    "改写成适合在中文求职简历资料库里做向量检索的规范中文查询，"
    "把英文/缩写换成对应中文说法。只输出改写后的查询，不要解释。\n\n问题：{q}"
)

HYDE_PROMPT = (
    "假设你在为候选人范睿峰的求职知识库补写资料。针对下面的问题，"
    "直接写一段 60~100 字的中文资料片段作为回答内容"
    "（具体细节可以合理虚构，风格与简历资料一致；不要标题、不要解释、不要说不知道）。\n\n问题：{q}"
)


def _llm_generate(prompt_template: str, q: str, max_tokens: int) -> str:
    r = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt_template.format(q=q)}],
    )
    return (r.choices[0].message.content or "").strip()


def rewrite_query(q: str) -> str:
    """query rewrite：口语问题 → 规范中文检索查询"""
    return _llm_generate(REWRITE_PROMPT, q, max_tokens=100) or q


def hyde_doc(q: str) -> str:
    """HyDE：先生成假想资料片段，用它代替原 query 去 embed"""
    return _llm_generate(HYDE_PROMPT, q, max_tokens=200) or q


def retrieve(query, k=5, use_rerank=False):
    """
    返回 [(meta, doc, similarity), ...]，按相关性从高到低。
    use_rerank=False: 纯 bi-encoder（原方案）
    use_rerank=True:  两阶段——bi-encoder 召回 top-RECALL_K，再 BGE-Reranker 重排取前 RERANK_TOP
    """
    emb = get_embedding(query)
    fetch_k = RECALL_K if use_rerank else k
    res = collection.query(
        query_embeddings=[emb], n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    base = list(zip(
        res["metadatas"][0], res["documents"][0], [1.0 - d for d in res["distances"][0]],
    ))  # [(meta, doc, sim), ...]

    if not use_rerank:
        return base[:k]

    # 两阶段：把 bi-encoder 召回的 RECALL_K 条交给 reranker 重排
    docs = [d for _, d, _ in base]
    ranked = rerank(query, docs, top_n=RERANK_TOP)  # [(orig_idx, score), ...]
    out = []
    for orig_idx, score in ranked:
        meta, doc, _ = base[orig_idx]
        out.append((meta, doc, score))  # 用 reranker 分当"相似度"展示
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
    {"q": "你现在 base 在哪里？",            "expect": {"profile.md", "faq.md"}},  # faq 补了口语变体条目，两处都是合法来源
    # ---- faq.md（HR 高频行为问题）----
    {"q": "你的三年职业规划是什么？",        "expect": {"faq.md"}},
    {"q": "你为什么转行做 AI？",             "expect": {"faq.md"}},
    {"q": "你的优点和缺点是什么？",          "expect": {"faq.md"}},
    {"q": "我们为什么应该录用你？",          "expect": {"faq.md"}},
    {"q": "你的期望薪资是多少？",            "expect": {"faq.md", "job-target.md"}},
    {"q": "你为什么从上一份工作离职？",      "expect": {"faq.md", "work-experience.md"}},
    {"q": "你这段空窗期在做什么？",          "expect": {"faq.md", "work-experience.md"}},
    {"q": "你的抗压能力怎么样？",            "expect": {"faq.md"}},
    {"q": "你能接受加班吗？",                "expect": {"faq.md"}},
    {"q": "你什么时候能入职？",              "expect": {"faq.md"}},
    {"q": "你有什么想问我的？",              "expect": {"faq.md"}},
    # ---- faq.md 口语变体（词汇桥接：口语问法 vs 语料书面措辞）----
    {"q": "你今年多大了？",                  "expect": {"profile.md", "faq.md"}},
    # ---- portfolio-deep.md（作品集工程深挖，2026-08-14 新增）----
    {"q": "作品集是怎么部署上线的？",        "expect": {"portfolio-deep.md", "projects.md"}},
    {"q": "你的 Agent 是怎么设计的？",       "expect": {"portfolio-deep.md"}},
    {"q": "检索质量是怎么评估的？",          "expect": {"portfolio-deep.md", "projects.md"}},
    {"q": "做过哪些失败的实验？",            "expect": {"portfolio-deep.md", "projects.md"}},  # projects.md 也记载了 reranker 负结果
    {"q": "项目踩过什么坑？",                "expect": {"portfolio-deep.md"}},
    {"q": "流式输出是怎么实现的？",          "expect": {"portfolio-deep.md"}},
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
    rebuild = "--rebuild" in sys.argv
    use_rerank = "--rerank" in sys.argv
    use_rewrite = "--rewrite" in sys.argv
    use_hyde = "--hyde" in sys.argv

    # 重建：删掉旧 chunk，用新切块策略重新入库（切换切块策略时用）
    if rebuild:
        print(f"重建模式：清空 collection 后用新切块重入库 ...")
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        collection_new = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
        )
        from mini_rag import RAGBot
        RAGBot("./data")  # 空库 → 触发 _ensure_embeddings 用新切块入库
        return  # 重建单独跑，不评估

    # 自举：本地库为空就先入库（复用 mini_rag 的入库逻辑）
    if collection.count() == 0:
        print(f"Chroma 为空（count=0），先调 BGE-M3 入库 ./data ...")
        from mini_rag import RAGBot
        RAGBot("./data")
    transform = None
    if use_rewrite and use_hyde:
        print("--rewrite 和 --hyde 互斥，请只选一个"); return
    if use_rewrite:
        transform = rewrite_query
    elif use_hyde:
        transform = hyde_doc

    modes = []
    if use_rerank:
        modes.append("bi-encoder + BGE-Reranker 两阶段")
    else:
        modes.append("bi-encoder 纯检索")
    if transform is rewrite_query:
        modes.append("query rewrite (DeepSeek)")
    elif transform is hyde_doc:
        modes.append("HyDE 假想文档 (DeepSeek)")
    mode = " + ".join(modes)
    print(f"知识库就绪：{collection.count()} chunks | 模式：{mode} | 评估 {len(GOLDEN)} 个问题\n")

    sums = {"h1": 0, "h3": 0, "h5": 0, "rr": 0.0}
    print(f"{'#':>2} {'H@1':>4} {'H@3':>4} {'H@5':>4} {'RR':>5}  问题")
    print("-" * 60)
    for i, item in enumerate(GOLDEN, 1):
        q = item["q"]
        transformed = transform(q) if transform is not None else None  # 检索前变换 query（实验路径）
        ret = retrieve(transformed or q, k=RERANK_TOP if use_rerank else 5, use_rerank=use_rerank)
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
        if transformed:
            print(f"    [变换] {transformed}")
        if verbose:
            for m, d, sim in ret[:3]:
                preview = d[:48].replace("\n", " ")
                print(f"      {m['source']}#{m['chunk_index']} sim={sim:.3f} | {preview}")
        if not h3:
            print(f"    ⚠ 未命中——可能需要调切块/检索策略")

    n = len(GOLDEN)
    print("\n" + "=" * 40)
    print(f"汇总 ({mode}, n={n})")
    print(f"  Hit@1 = {sums['h1']/n:.1%}   （top1 就对）")
    print(f"  Hit@3 = {sums['h3']/n:.1%}   （top3 有命中，线上用的是 top3）")
    print(f"  Hit@5 = {sums['h5']/n:.1%}")
    print(f"  MRR   = {sums['rr']/n:.3f}   （越接近 1 越好，1=每次 top1 就命中）")
    print("=" * 40)


if __name__ == "__main__":
    main()

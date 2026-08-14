# eval_generation.py
# 生成忠实度评估（LLM-as-judge）
#
# 原理：生成回答 → 让第二个 LLM 当裁判，逐句检查回答里每个事实陈述
#       是否都能在参考资料里找到依据。
# 指标：忠实度 = 无编造陈述的问题占比；回答率 = 实质回答（非拒答）的问题占比。
#       只看忠实度会被「全拒答」刷满（拒答的回答天然无编造），必须两个一起看。
#       同时列出被判定为「无依据」的句子。
#
# 两种模式：
#   python eval_generation.py                 # 默认：纯 RAG 链路（检索 + 生成，本地复刻）
#   python eval_generation.py --production    # 生产：直测 LangGraph Agent 的 /ask 回答，
#                                             # 裁判对照完整知识库（须在 service 容器内跑）
#
# 用法：
#   纯 RAG 模式任意环境都能跑；--production 模式要在 backend 服务容器内跑
#   （docker-compose exec backend python eval_generation.py --production），
#   因为它直接请求 http://localhost:8000/ask。
#
# 成本：N 个问题 ×（1 次生成 + 1 次裁判）次 DeepSeek 调用，量级很小。

import argparse
import json
import os
import re
import uuid

import chromadb
import requests
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    CHAT_MODEL,
    SILICONFLOW_BASE_URL,
    DEEPSEEK_BASE_URL,
    GITHUB_USERNAME,
)

load_dotenv()

embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=SILICONFLOW_BASE_URL,
)
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_BASE_URL,
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
)

# 12 个 HR 高频问题，覆盖不同资料源 + 容易引发编造的「未来计划/数字/细节」类问题
QUESTIONS = [
    "你会什么编程语言？",
    "你做过哪些项目？",
    "你的上一份工作是什么？",
    "你的求职方向是什么？",
    "你的期望薪资是多少？",
    "你的三年职业规划是什么？",
    "你的网站地址是什么？",
    "怎么联系你？",
    "你的 GitHub 上有哪些项目？",
    "你这段空窗期在做什么？",
    "你的 RAG 知识库现在有多少个 chunk？",
    "你接下来有什么学习计划？",
]

# 与生产提示词保持同构（改生产 SYSTEM_PROMPT 时这里同步改，评估的就是真实链路）
RAG_PROMPT = """你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 只基于下面【参考资料】回答，不要凭空编造
2. 如果资料里没有相关信息，直接说"这方面资料我暂时没有详细说明"
3. 不要编造经历、技能或项目
4. 语气专业、简洁、自信
5. 用第一人称回答（"我"指代范睿峰）
6. 资料里没有明确写的未来计划、细节或数字，禁止自行补充，不确定就直接说"暂时没有详细说明"

【参考资料】
{context}

【HR 的问题】
{question}
"""

JUDGE_PROMPT = """你是严格的评估裁判。下面是一个 AI 求职助手对 HR 问题的回答，以及它参考的资料。
请逐句检查回答：每个事实性陈述（数字、时间、技能、经历、计划、链接、联系方式）是否都能在参考资料里找到依据？
只输出 JSON（不要输出任何其他内容、不要用 markdown 代码块）：
{{"faithful": 1, "unsupported": ["无依据的陈述1", "无依据的陈述2"]}}
faithful=1 表示所有事实陈述都有资料依据；faithful=0 表示存在编造或无依据的陈述，把具体句子列进 unsupported（没有就输出空数组）。
注意：像"很高兴认识你"这类礼貌用语不算事实陈述，不要判为编造。

【参考资料】
{context}

【问题】
{question}

【回答】
{answer}
"""

JUDGE_PROMPT_PRODUCTION = """你是严格的评估裁判。下面是一个 AI 求职助手对 HR 问题的回答。
参考资料是候选人范睿峰的资料，包含两部分：
① 完整知识库（所有资料文件的全部内容：个人简介、技能、项目、工作经历、求职意向、HR 高频问答）
② 实时 GitHub API 数据（仅当下方确实出现该节时才作为依据；获取失败会自动省略，省略时对 GitHub 类陈述判"无依据"）
请逐句检查回答：每个事实性陈述（数字、时间、技能、经历、计划、链接、联系方式）是否都能在参考资料里找到依据？
只输出 JSON（不要输出任何其他内容、不要用 markdown 代码块）：
{{"faithful": 1, "unsupported": ["无依据的陈述1", "无依据的陈述2"]}}
faithful=1 表示所有事实陈述都有资料依据；faithful=0 表示存在编造或无依据的陈述，把具体句子列进 unsupported（没有就输出空数组）。
注意：像"很高兴认识你"这类礼貌用语不算事实陈述，不要判为编造。

【参考资料】
{context}

【问题】
{question}

【回答】
{answer}
"""


def live_github_context():
    """
    实时 GitHub 数据（与生产 Agent 的 GitHub 工具同源），供裁判核对 GitHub 类回答。
    必须检查 HTTP 状态码：未认证 GitHub API 限流 403 时，响应体是错误 JSON（无
    public_repos 等字段），不检查就会变成 public_repos=None 的垃圾数据、还被裁判
    提示词钦定为"可信依据"。任何失败都返回 ""（裁判侧该节自动省略）。
    """
    try:
        resp_u = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}", timeout=6
        )
        resp_r = requests.get(
            f"https://api.github.com/users/{GITHUB_USERNAME}/repos?sort=updated&per_page=5",
            timeout=6,
        )
        if resp_u.status_code != 200 or resp_r.status_code != 200:
            return ""
        u, rs = resp_u.json(), resp_r.json()
        if not isinstance(u, dict) or "login" not in u or not isinstance(rs, list):
            return ""
        parts = [
            f"GitHub 用户 {u.get('login', GITHUB_USERNAME)}: "
            f"public_repos={u.get('public_repos')}, followers={u.get('followers')}"
        ]
        for r in rs:
            parts.append(
                f"仓库 {r.get('name')}: 描述={r.get('description') or '无'}, "
                f"语言={r.get('language') or '未知'}"
            )
        return "\n".join(parts)
    except Exception:
        return ""


def retrieve(query, k=3):
    emb = embed_client.embeddings.create(model=EMBED_MODEL, input=query)
    res = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=k,
        include=["documents"],
    )
    return res["documents"][0] if res["documents"] else []


def kb_dump():
    docs = collection.get(include=["documents"])["documents"] or []
    return "\n\n".join(docs)


def generate(question, context):
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def ask_production(question, session_id):
    resp = requests.post(
        "http://localhost:8000/ask",
        json={"question": question, "session_id": session_id},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def judge(question, answer, context, production=False):
    template = JUDGE_PROMPT_PRODUCTION if production else JUDGE_PROMPT
    prompt = template.format(context=context, question=question, answer=answer)
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content
    # 裁判偶尔会包一层 ```json 围栏，剥掉再解析
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {
        "faithful": 0,
        "unsupported": [f"裁判输出无法解析: {text[:80]}"],
    }


DEFLECT_RE = re.compile(r"暂时没有详细说明|没有详细说明|资料里没有|没有相关")


def is_deflection(answer: str) -> bool:
    """
    拒答判定（启发式）：按句拆分，剔除含拒答措辞的句子后，剩余实质内容不足 10 字即判拒答。
    （拒答的回答天然无编造，只报忠实度会被「全拒答」刷满，所以必须伴生报告回答率。
      不能用"整条短+含措辞"判——实质回答末尾带一句"其他方面暂时没有详细说明"会被误杀。）
    """
    a = (answer or "").strip()
    if not a:
        return True
    sentences = re.split(r"(?<=[。！？!?\n])", a)
    substantive = "".join(s for s in sentences if not DEFLECT_RE.search(s))
    core = re.sub(r"[\s，。、！？,.!?；;:：\"'（）()]+", "", substantive)
    return len(core) < 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--production",
        action="store_true",
        help="直测生产 LangGraph Agent 的 /ask 回答（裁判对照完整知识库）；默认评估纯 RAG 链路",
    )
    args = ap.parse_args()

    mode = "生产 Agent (/ask)" if args.production else "纯 RAG 链路"
    print(f"开始评估 {len(QUESTIONS)} 个问题 | 模式: {mode} | 每个问题 1 次生成 + 1 次裁判...\n")

    # --production 用随机 session 前缀：固定 eval-1..12 会写进生产 MemorySaver，
    # 既污染真实访客对话记忆，也让重跑结果被上一次的对话历史污染（不再独立）
    run_id = uuid.uuid4().hex[:8]

    # 实时 GitHub 数据整场只拉一次（每题都拉会烧未认证 API 的 60 次/时配额，
    # 那是和线上真实访客共享的出口 IP 配额）
    live = live_github_context() if args.production else ""
    if args.production and not live:
        print("⚠️ 实时 GitHub 数据获取失败（可能限流）。GitHub 类回答会被裁判判为无依据，结果仅供参考。\n")

    faithful_count = 0
    answered_count = 0
    faithful_answered = 0
    deflected = []
    for i, q in enumerate(QUESTIONS, 1):
        if args.production:
            answer = ask_production(q, f"eval-{run_id}-{i}")
            context = kb_dump()
            if live:
                context += "\n\n【实时 GitHub API 数据】\n" + live
        else:
            docs = retrieve(q)
            context = "\n\n".join(docs)
            answer = generate(q, context)
        verdict = judge(q, answer, context, production=args.production)
        faithful = 1 if verdict.get("faithful") == 1 else 0
        faithful_count += faithful
        answered = not is_deflection(answer)
        answered_count += answered
        if answered and faithful:
            faithful_answered += 1
        if not answered:
            deflected.append(q)
        status = "✅ 忠实" if faithful else "❌ 有编造"
        deflect_mark = "" if answered else "｜🚫 拒答"
        print(f"[{i}/{len(QUESTIONS)}] {status}{deflect_mark} | {q}")
        if not faithful:
            for u in verdict.get("unsupported", []):
                print(f"      ⚠️ 无依据陈述: {u[:120]}")
        print()
    n = len(QUESTIONS)
    answer_rate = answered_count / n * 100
    print("=" * 56)
    print(f"回答率: {answered_count}/{n} = {answer_rate:.1f}%（拒答不贡献信息，忠实度必须结合回答率看）")
    if deflected:
        for q in deflected:
            print(f"      🚫 拒答: {q}")
    print(f"忠实度: {faithful_count}/{n}（n={n} 小样本，勿单看百分比）")
    if answered_count:
        print(f"已回答问题中的忠实率: {faithful_answered}/{answered_count}")
    print("(忠实 = 回答中所有事实陈述都能在参考资料里找到依据)")


if __name__ == "__main__":
    main()

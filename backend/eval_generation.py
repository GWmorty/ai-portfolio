# eval_generation.py
# 生成忠实度评估（LLM-as-judge）
#
# 原理：检索 → 生成 → 让第二个 LLM 当裁判，逐句检查回答里每个事实陈述
#       是否都能在【检索到的资料】里找到依据。
# 指标：忠实度 = 无编造陈述的问题占比；同时列出被判定为「无依据」的句子。
#
# 适用范围说明：这里评估的是「RAG 检索 + DeepSeek 生成」这条链路的忠实度
# （与 server.py/miniRGA 同款 prompt），不是 LangGraph Agent 的完整工具调用行为。
#
# 用法（backend 目录下，容器内 /app 同理）：
#   python eval_generation.py
#
# 成本：N 个问题 ×（1 次生成 + 1 次裁判）次 DeepSeek 调用，量级很小。

import json
import os
import re

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_PATH = os.path.expanduser("~/.ai_portfolio/chroma_db")
COLLECTION_NAME = "knowledge_base"

embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"},
)

# 10 个 HR 高频问题，覆盖不同资料源 + 容易引发编造的「未来计划/数字」类问题
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
]

# 与生产提示词保持同构（改生产 SYSTEM_PROMPT 时这里同步改，评估的就是真实链路）
RAG_PROMPT = """你是一个 AI 求职助手，代表候选人范睿峰回答 HR 的问题。

【重要规则】
1. 只基于下面【参考资料】回答，不要凭空编造
2. 如果资料里没有相关信息，直接说"这方面资料我暂时没有详细说明"
3. 不要编造经历、技能或项目
4. 语气专业、简洁、自信
5. 用第一人称回答（"我"指代范睿峰）

【参考资料】
{context}

【HR 的问题】
{question}
"""

JUDGE_PROMPT = """你是严格的评估裁判。下面是一个 AI 求职助手对 HR 问题的回答，以及它检索到的参考资料。
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


def retrieve(query, k=3):
    emb = embed_client.embeddings.create(model="BAAI/bge-m3", input=query)
    res = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=k,
        include=["documents"],
    )
    return res["documents"][0] if res["documents"] else []


def generate(question, context):
    prompt = RAG_PROMPT.format(context=context, question=question)
    resp = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content


def judge(question, answer, context):
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    resp = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content
    # 裁判偶尔会包一层 ```json 围栏，剥掉再解析
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"faithful": 0, "unsupported": [f"裁判输出无法解析: {text[:80]}"], "_raw": text}


def main():
    faithful_count = 0
    print(f"开始评估 {len(QUESTIONS)} 个问题（每个问题 1 次生成 + 1 次裁判）...\n")
    for i, q in enumerate(QUESTIONS, 1):
        docs = retrieve(q)
        context = "\n\n".join(docs)
        answer = generate(q, context)
        verdict = judge(q, answer, context)
        faithful = 1 if verdict.get("faithful") == 1 else 0
        faithful_count += faithful
        status = "✅ 忠实" if faithful else "❌ 有编造"
        print(f"[{i}/{len(QUESTIONS)}] {status} | {q}")
        if not faithful:
            for u in verdict.get("unsupported", []):
                print(f"      ⚠️ 无依据陈述: {u[:120]}")
        print()
    rate = faithful_count / len(QUESTIONS) * 100
    print("=" * 56)
    print(f"忠实度: {faithful_count}/{len(QUESTIONS)} = {rate:.1f}%")
    print("(忠实 = 回答中所有事实陈述都能在检索到的资料里找到依据)")


if __name__ == "__main__":
    main()

"use client";

import { useState } from "react";

export default function DevLogSection() {
  const [openIdx, setOpenIdx] = useState(null);

  const issues = [
    {
      tag: "后端",
      title: "流式输出全程 500 报错",
      problem: "LangGraph 的 /ask 一请求就 500。",
      cause: "async def 节点用了同步 invoke()，LangGraph 1.x 禁止这种混用。",
      fix: "改成 await ainvoke()。",
      lesson: "框架升级（>= 依赖）会静默改变 API 约束，锁版本前要读 changelog。",
    },
    {
      tag: "Agent",
      title: "流式回答里泄漏了一堆原始资料",
      problem: "AI 回答开头突然蹦出 ## 编程技能 ### Python... 这种 markdown 原文。",
      cause: "stream_mode=messages 会流出所有 message，包括工具检索回来的 ToolMessage（原文），我没过滤。",
      fix: "过滤条件加 isinstance(msg, AIMessage)，只透出 LLM 的回答。",
      lesson: "agent 的「思考过程」和「最终输出」要分开，不能一股脑给前端。",
    },
    {
      tag: "部署",
      title: "线上流式只吐 [DONE]，本地却正常",
      problem: "本地逐字流式，部署到 Docker 后线上只剩个 [DONE]。",
      cause: "requirements.txt 用 >=，本地和容器装到不同版本，astream 吐 AIMessageChunk 还是整条 AIMessage 行为不同。",
      fix: "过滤规则兼容两种消息类型；根因是按容器 pip freeze 把依赖全 == 锁死。",
      lesson: ">= 是多环境漂移的根源；本地能跑 ≠ 线上能跑。",
    },
    {
      tag: "RAG",
      title: "AI 全答 GitHub 信息，个人技能答不出",
      problem: "问你会什么，AI 不从知识库答，全靠 GitHub 工具硬撑。",
      cause: "ChromaDB 是空的——server 只读不写，入库逻辑在另一个脚本里、从没在容器跑过。",
      fix: "容器内手动跑入库；入库后必须重启容器（Chroma 跨进程不刷新）。",
      lesson: "数据初始化是独立步骤，别假设代码跑起来数据就在。",
    },
    {
      tag: "RAG",
      title: "reranker 加了反而检索更差",
      problem: "听说 reranker 提精度，加上一测 Hit@1 从 87% 掉到 81%。",
      cause: "数据集太小（22 个短 chunk），bi-encoder 已逼近最优，reranker 徒增噪声。",
      fix: "数据说话，没上线。",
      lesson: "先进技术 ≠ 更好，得用 eval 量化验证，不能迷信。",
    },
    {
      tag: "前端",
      title: "流式输出一阵阵卡顿",
      problem: "逐字效果像一次吐十个、停一下、再吐五个。",
      cause: "DeepSeek API + TCP 攒批，token 是一阵阵到达的。",
      fix: "前端加队列 + 匀速定时器，渲染节奏与到达节奏解耦。",
      lesson: "上游不稳就加缓冲，producer-consumer 解耦是通用模式。",
    },
    {
      tag: "前端",
      title: "中途打断发新问题，旧回答混进新回答",
      problem: "吐到一半点新问题，新回答开头出现上一题的尾巴。",
      cause: "abort 后 JS 的 async reader 还在跑最后一轮，把迟到 token 塞进新队列。",
      fix: "每次发问分配 runId，token 入队和渲染都校验 id，旧流迟到 token 一律丢。",
      lesson: "abort 不能瞬间停掉一切，异步竞态要用逻辑标记区分新旧。",
    },
    {
      tag: "Agent",
      title: "答案开头冒英文「Let me search...」",
      problem: "问问题后，AI 回答开头突然蹦出一段英文「Let me search for information about...」，然后才是中文答案。",
      cause: "DeepSeek 决定调工具时会先吐一段英文独白；每个流式 chunk 都是 AIMessageChunk 且 tool_calls=False（tool_calls 要等这条消息流完才聚合），被旧过滤当成最终答案透传了。",
      fix: "prompt 治本（要求调工具时不输出文字、始终用中文）+ 后端兜底（缓冲工具前的 AIMessageChunk，遇 ToolMessage 判定为前奏丢弃，工具后的才是真答案）。",
      lesson: "流式过滤光看「单个 chunk 有没有 tool_calls」不够，因为 tool_calls 是延迟聚合的——反直觉的时序陷阱。",
    },
    {
      tag: "部署",
      title: "国内 HR 打不开网站",
      problem: "换 DuckDNS 域名后国内要翻墙。",
      cause: "DuckDNS 在国内被 DNS 污染。",
      fix: "换成自有域名 szds.site，配 Let's Encrypt HTTPS，国内可直接访问。",
      lesson: "免费服务有隐藏成本；给国内用户用，得用国内能解析的域名。",
    },
  ];

  const tagStyles = {
    后端: "bg-blue-100 text-blue-800 border border-blue-200",
    Agent: "bg-purple-100 text-purple-800 border border-purple-200",
    RAG: "bg-emerald-100 text-emerald-800 border border-emerald-200",
    前端: "bg-amber-100 text-amber-800 border border-amber-200",
    部署: "bg-zinc-200 text-zinc-800 border border-zinc-300",
  };

  return (
    <section id="devlog" className="py-24 px-6 bg-white border-t border-zinc-100">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <p className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
            Dev Log
          </p>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-zinc-900">
            开发踩坑实录
          </h2>
          <p className="mt-3 text-zinc-600">
            开发过程中真实遇到的问题、定位到的根因、最后的解法，以及每一条留下的教训。比起光鲜的成品，这些坑更能反映真实的工程思考过程。
          </p>
        </div>

        <div className="space-y-2">
          {issues.map((issue, i) => {
            const isOpen = openIdx === i;
            return (
              <article
                key={i}
                className={`rounded-xl border transition-all ${
                  isOpen
                    ? "border-zinc-300 bg-zinc-50 shadow-sm"
                    : "border-zinc-200 bg-white hover:border-zinc-300"
                }`}
              >
                {/* 标题行（可点击）*/}
                <button
                  onClick={() => setOpenIdx(isOpen ? null : i)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left"
                >
                  <span className={`flex-shrink-0 px-2 py-0.5 text-xs font-medium rounded-full ${tagStyles[issue.tag]}`}>
                    {issue.tag}
                  </span>
                  <span className={`flex-1 text-sm ${isOpen ? "font-semibold text-zinc-900" : "font-medium text-zinc-800"}`}>
                    {issue.title}
                  </span>
                  <span className={`flex-shrink-0 text-zinc-400 transition-transform ${isOpen ? "rotate-90" : ""}`}>
                    ▶
                  </span>
                </button>

                {/* 详情区（展开时才渲染）*/}
                {isOpen && (
                  <div className="px-4 pb-4 pt-1">
                    <dl className="space-y-2 text-sm">
                      <div className="flex gap-2">
                        <dt className="flex-shrink-0 w-12 text-zinc-400">现象</dt>
                        <dd className="text-zinc-700">{issue.problem}</dd>
                      </div>
                      <div className="flex gap-2">
                        <dt className="flex-shrink-0 w-12 text-zinc-400">根因</dt>
                        <dd className="text-zinc-700">{issue.cause}</dd>
                      </div>
                      <div className="flex gap-2">
                        <dt className="flex-shrink-0 w-12 text-zinc-400">解决</dt>
                        <dd className="text-zinc-700">{issue.fix}</dd>
                      </div>
                    </dl>
                    <div className="mt-3 pt-3 border-t border-zinc-200/70 flex gap-2 text-sm">
                      <span className="flex-shrink-0 text-zinc-400">💡 教训</span>
                      <span className="text-zinc-900 font-medium">{issue.lesson}</span>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

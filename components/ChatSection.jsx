"use client";

import { useState, useRef, useEffect } from "react";

// 快捷问题（HR 一键点击填入输入框）
const QUICK_QUESTIONS = [
  "你会什么编程语言？",
  "做过什么 AI 项目？",
  "求职方向是什么？",
  "怎么联系你？",
];

// 初始欢迎语
const WELCOME_MESSAGE = {
  role: "assistant",
  content:
    "你好！我是范睿峰的 AI 求职助手。想了解我的技能、项目、工作经历或求职方向吗？直接问吧！",
};

// 后端兜底联系方式（流式出错时显示）
const FALLBACK_TEXT =
  "抱歉，AI 助手暂时无法响应。请直接联系范睿峰：\n\n📞 18021080437\n📧 2413824669@qq.com";

export default function ChatSection() {
  // 对话历史
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  // 输入框内容
  const [input, setInput] = useState("");
  // 是否正在等待 / 接收 AI 回复
  const [loading, setLoading] = useState(false);
  // ⭐ 每个访客独立的 session_id（支持多轮对话记忆）
  const [sessionId] = useState(
    () => `hr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );
  // 自动滚动到底部
  const scrollRef = useRef(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);
  // token 渲染队列：后端"一阵阵"到的 token 进队列，定时器匀速弹出，视觉丝滑
  const tokenQueueRef = useRef([]);
  const flushTimerRef = useRef(null);
  const streamEndedRef = useRef(false);  // 流是否已结束（结束后定时器把队列弹空就停）
  const abortRef = useRef(null);  // 当前 fetch 的 AbortController，用于中途发新问题时中断旧流
  const runIdRef = useRef(0);  // 每次发问自增，用于区分"当前有效流"——旧流迟到 token 因 runId 不匹配被丢

  // 中断正在进行的流：直接丢弃队列剩余、abort 后端、停定时器。
  // （不追加剩余到消息——状态异步更新下"取最后一条消息"会错位写到新消息开头，丢弃更干净）
  const abortStream = () => {
    if (abortRef.current) { try { abortRef.current.abort(); } catch {} abortRef.current = null; }
    clearInterval(flushTimerRef.current);
    tokenQueueRef.current = [];
    streamEndedRef.current = true;
  };

  // 发送消息（流式：逐字渲染 AI 回复）
  const sendMessage = async (question = null) => {
    const text = (question || input).trim();
    if (!text) return;

    // 若有正在进行的流，先中断它（清队列、停定时器、abort 旧 fetch）
    if (loading) abortStream();
    // 本次发问的唯一 id：旧流迟到的 token 因 runId 不匹配会被丢弃，避免混进新答复
    const myRunId = ++runIdRef.current;

    // 1. 添加用户消息 + 一条空的 AI 消息（后续 token / steps 往里填）
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", steps: [] },
    ]);
    setInput("");
    setLoading(true);

    const API_URL =
      process.env.NODE_ENV === "development"
        ? "http://localhost:8000/ask/stream"
        : "/api/ask/stream";

    try {
      abortRef.current = new AbortController();
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, session_id: sessionId }),
        signal: abortRef.current.signal,
      });
      if (!response.ok) throw new Error(`API 错误: ${response.status}`);

      // 匀速弹出定时器：每 7ms 弹 1 个字符（约 140 字/秒）。
      // 关键：渲染速度 < 生成速度，让队列保持积压，视觉匀速逐字，
      // 不受 DeepSeek 攒批影响。流结束后继续弹，直到队列空了才停。
      // runId 校验：只有当前流的 token 才渲染，旧流迟到的 token 被丢
      tokenQueueRef.current = [];
      streamEndedRef.current = false;
      clearInterval(flushTimerRef.current);
      flushTimerRef.current = setInterval(() => {
        if (runIdRef.current !== myRunId) { clearInterval(flushTimerRef.current); return; }
        if (tokenQueueRef.current.length === 0) {
          if (streamEndedRef.current) { clearInterval(flushTimerRef.current); }
          return;
        }
        const joined = tokenQueueRef.current.splice(0).join("");
        const out = joined.slice(0, 1);
        if (joined.length > 1) tokenQueueRef.current.unshift(joined.slice(1));
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + out };
          }
          return next;
        });
      }, 7);

      // 2. 用 reader 读取 SSE 流
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE 事件之间用空行分隔；最后一段可能不完整，留着下次拼
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const evt of events) {
          const line = evt.trim();
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const obj = JSON.parse(payload);
            if (obj.type === "token" && obj.content) {
              // runId 校验：旧流 abort 后迟到的 token 直接丢，不混进新队列
              if (runIdRef.current !== myRunId) continue;
              tokenQueueRef.current.push(obj.content);
            } else if (obj.type === "tool_start" || obj.type === "tool_end") {
              // runId 校验：旧流迟到的工具事件也丢
              if (runIdRef.current !== myRunId) continue;
              // 工具事件立即应用（不缓冲）
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (obj.type === "tool_start") {
                  const steps = [...(last.steps || [])];
                  steps.push({ tool: obj.tool, label: obj.label, query: obj.query, status: "running" });
                  next[next.length - 1] = { ...last, steps };
                } else if (obj.type === "tool_end") {
                  const steps = [...(last.steps || [])];
                  for (let i = steps.length - 1; i >= 0; i--) {
                    if (steps[i].status === "running" && steps[i].tool === obj.tool) {
                      steps[i] = { ...steps[i], status: "done", preview: obj.preview };
                      break;
                    }
                  }
                  next[next.length - 1] = { ...last, steps };
                }
                return next;
              });
            }
          } catch {
            // 忽略半截 JSON
          }
        }
      }
    } catch (error) {
      // 中途被新问题中断（AbortError）→ 不填兜底文案，静默退出（被 abortStream 已清理）
      if (error.name === "AbortError") return;
      // 网络失败 / 后端没启动 → 在那条空 AI 消息里填兜底文案
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last.role === "assistant" && last.content === "") {
          return [...prev.slice(0, -1), { role: "assistant", content: FALLBACK_TEXT }];
        }
        return [...prev, { role: "assistant", content: FALLBACK_TEXT }];
      });
    } finally {
      // 不一次性 flush 剩余（会蹦一大段破坏匀速），只标记流结束；
      // 定时器会把队列里剩余 token 按 10ms/字继续匀速弹完，弹空了自动停。
      // 若是中途被 abort，abortStream 已置 streamEnded=true，这里无副作用。
      streamEndedRef.current = true;
      setLoading(false);
    }
  };

  // 按 Enter 发送（Shift+Enter 换行）
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // 组件卸载时清定时器，避免泄漏
  useEffect(() => () => clearInterval(flushTimerRef.current), []);

  return (
    <section
      id="chat"
      className="min-h-screen flex flex-col items-center justify-center px-4 py-20 bg-gradient-to-b from-gray-50 to-white"
    >
      <div className="max-w-3xl w-full">
        {/* 标题 */}
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
            和我的 AI 助手对话
          </h2>
          <p className="text-gray-600 text-base md:text-lg">
            HR 直接提问，AI 基于我的真实资料回答（不编造）
          </p>
        </div>

        {/* 对话容器 */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-200 h-[520px] flex flex-col overflow-hidden">
          {/* 消息列表（可滚动） */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
            {messages.map((msg, i) => {
              const isLast = i === messages.length - 1;
              const showTypingDots =
                msg.role === "assistant" && isLast && loading && msg.content === "";
              return (
                <div
                  key={i}
                  className={`flex flex-col ${
                    msg.role === "user" ? "items-end" : "items-start"
                  }`}
                >
                  {/* agent 思考过程（工具步骤）——独立卡片，宽度限制 85% */}
                  {msg.role === "assistant" && msg.steps && msg.steps.length > 0 && (
                    <div className="mb-1.5 max-w-[85%] rounded-lg bg-gray-50 border border-gray-200 px-3 py-2 text-xs text-gray-500">
                      <div className="flex items-center gap-1 mb-1 font-medium text-gray-400">
                        <span>🤖</span>
                        <span>思考过程</span>
                      </div>
                      <div className="space-y-1">
                        {msg.steps.map((s, si) => (
                          <div key={si} className="flex items-center gap-1.5">
                            <span className={s.status === "running" ? "animate-pulse" : ""}>
                              {s.status === "running" ? "⏳" : "✓"}
                            </span>
                            <span>
                              {s.label}
                              {s.query ? `：${s.query}` : ""}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm md:text-base break-words ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white rounded-br-sm"
                        : "bg-gray-100 text-gray-800 rounded-bl-sm"
                    }`}
                  >
                    {showTypingDots ? (
                      <div className="flex gap-1.5 px-1 py-1">
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
                        <span
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.15s" }}
                        ></span>
                        <span
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.3s" }}
                        ></span>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* 输入区域 */}
          <div className="border-t border-gray-200 p-3 md:p-4 bg-white">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                maxLength={500}
                placeholder="问问我的技能、项目、经历..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim()}
                className="px-4 md:px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
              >
                发送
              </button>
            </div>
            <p className="text-xs text-gray-400 mt-2 text-center">
              按 Enter 发送 · AI 基于真实资料回答，不会编造
            </p>
          </div>
        </div>

        {/* 快捷问题按钮 */}
        <div className="mt-6 flex flex-wrap gap-2 justify-center">
          {QUICK_QUESTIONS.map((q) => (
            <button
              key={q}
              onClick={() => sendMessage(q)}
              className="px-3 py-1.5 text-sm text-gray-800 bg-white border border-gray-300 rounded-full hover:bg-blue-50 hover:border-blue-300 transition-colors"
            >
              {q}
            </button>
          ))}
        </div>

        {/* 技术说明（小字） */}
        <p className="text-center text-xs text-gray-400 mt-6">
          Powered by LangGraph + BGE-M3 + Chroma + DeepSeek API（流式 SSE）
        </p>
      </div>
    </section>
  );
}

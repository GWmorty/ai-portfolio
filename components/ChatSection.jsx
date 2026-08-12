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

  // 发送消息（流式：逐字渲染 AI 回复）
  const sendMessage = async (question = null) => {
    const text = (question || input).trim();
    if (!text || loading) return;

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
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, session_id: sessionId }),
      });
      if (!response.ok) throw new Error(`API 错误: ${response.status}`);

      // 匀速弹出定时器：每 22ms 把队列里所有 token 合并追加一次（视觉丝滑）
      tokenQueueRef.current = [];
      clearInterval(flushTimerRef.current);
      flushTimerRef.current = setInterval(() => {
        if (tokenQueueRef.current.length === 0) return;
        const chunk = tokenQueueRef.current.splice(0, tokenQueueRef.current.length).join("");
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + chunk };
          }
          return next;
        });
      }, 22);

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
              // token 不直接渲染，入队列；定时器匀速弹出（避免后端"一阵阵"导致卡顿）
              tokenQueueRef.current.push(obj.content);
            } else if (obj.type === "tool_start" || obj.type === "tool_end") {
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
      // 网络失败 / 后端没启动 → 在那条空 AI 消息里填兜底文案
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last.role === "assistant" && last.content === "") {
          return [...prev.slice(0, -1), { role: "assistant", content: FALLBACK_TEXT }];
        }
        return [...prev, { role: "assistant", content: FALLBACK_TEXT }];
      });
    } finally {
      // 停定时器，把队列里没弹完的 token 一次性 flush（别丢）
      clearInterval(flushTimerRef.current);
      if (tokenQueueRef.current.length > 0) {
        const rest = tokenQueueRef.current.splice(0).join("");
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = { ...last, content: last.content + rest };
          }
          return next;
        });
      }
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
                placeholder="问问我的技能、项目、经历..."
                disabled={loading}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
              />
              <button
                onClick={() => sendMessage()}
                disabled={loading || !input.trim()}
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
              disabled={loading}
              className="px-3 py-1.5 text-sm text-gray-800 bg-white border border-gray-300 rounded-full hover:bg-blue-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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

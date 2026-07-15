"use client";

import { useState } from "react";

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

export default function ChatSection() {
  // 对话历史
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  // 输入框内容
  const [input, setInput] = useState("");
  // 是否正在等待 AI 回复
  const [loading, setLoading] = useState(false);
  // ⭐ 每个访客独立的 session_id（支持多轮对话记忆）
  const [sessionId] = useState(
    () => `hr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );

  // 发送消息
  const sendMessage = async (question = null) => {
    // 没输入或正在 loading，忽略
    const text = (question || input).trim();
    if (!text || loading) return;

    // 1. 添加用户消息
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      // 2. 调流式端点（SSE 逐字输出，像 ChatGPT 打字机效果）
      const API_URL =
        process.env.NODE_ENV === "development"
          ? "http://localhost:8000/ask/stream"
          : "/api/ask/stream";

      // 先添加空的 AI 消息（等待逐字填充）
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, session_id: sessionId }),
      });

      if (!response.ok) throw new Error(`API 错误: ${response.status}`);

      // ⭐ 流式读取：逐字追加到 AI 消息
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ") && !line.includes("[DONE]")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.content) {
                aiText += data.content;
                // 逐字更新最后一条消息
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    role: "assistant",
                    content: aiText,
                  };
                  return updated;
                });
              }
            } catch (e) {}
          }
        }
      }
    } catch (error) {
      // 网络失败 / 后端没启动 → 显示联系方式
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "抱歉，AI 助手暂时无法响应。请直接联系范睿峰：\n\n📞 18021080437\n📧 2413824669@qq.com",
        },
      ]);
    } finally {
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
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm md:text-base ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-br-sm"
                      : "bg-gray-100 text-gray-800 rounded-bl-sm"
                  }`}
                >
                  {/* whitespace-pre-wrap 保留换行符 */}
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                </div>
              </div>
            ))}

            {/* Loading 动画（三个跳动小圆点） */}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-4">
                  <div className="flex gap-1.5">
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
                </div>
              </div>
            )}
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
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50"
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
              className="px-3 py-1.5 text-sm bg-white border border-gray-300 rounded-full hover:bg-blue-50 hover:border-blue-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {q}
            </button>
          ))}
        </div>

        {/* 技术说明（小字） */}
        <p className="text-center text-xs text-gray-400 mt-6">
          Powered by FastAPI + BGE-M3 + Chroma + DeepSeek API
        </p>
      </div>
    </section>
  );
}

import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "范睿峰 · AI 求职作品集",
  description: "用 AI 构建产品，记录学习与成长。Python、Next.js、大模型 API 实战项目。",
  openGraph: {
    title: "范睿峰 · AI 求职作品集",
    description:
      "LangGraph Agent + RAG + FastAPI + Next.js 全栈实战的 AI 方向求职作品集",
    url: "https://szds.site",
    siteName: "范睿峰 · AI 求职作品集",
    locale: "zh_CN",
    type: "website",
    images: ["https://szds.site/og.png"],
  },
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="zh-CN"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

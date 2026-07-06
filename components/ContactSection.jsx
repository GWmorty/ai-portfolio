export default function ContactSection() {
  return (
    <section id="contact" className="py-24 px-6 bg-zinc-900 border-t border-zinc-800">
      <div className="max-w-3xl mx-auto text-center">
        <p className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
          Contact
        </p>
        <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-white">
          联系我
        </h2>
        <p className="mt-4 text-zinc-400 leading-relaxed">
          如果你正在招 AI 方向的候选人，或者想交流学习路径，欢迎联系。
        </p>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <a
            href="mailto:your-email@example.com"
            className="inline-flex items-center px-6 py-3 rounded-full bg-white text-zinc-900 font-medium hover:bg-zinc-100 transition-colors"
          >
            ✉ 发邮件给我
          </a>
          <a
            href="https://github.com/your-username"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center px-6 py-3 rounded-full border border-zinc-700 text-zinc-200 font-medium hover:border-white hover:text-white transition-colors"
          >
            GitHub
          </a>
        </div>

        <footer className="mt-20 pt-8 border-t border-zinc-800 text-sm text-zinc-500">
          <p>© 2026 范睿峰 · 用 Next.js + Tailwind 构建</p>
        </footer>
      </div>
    </section>
  );
}
export default function HeroSection() {
  return (
    <section className="min-h-screen flex items-center justify-center px-6 py-20 bg-gradient-to-br from-zinc-50 via-white to-zinc-100">
      <div className="max-w-3xl mx-auto text-center">
        
        <p className="inline-block px-3 py-1 text-xs font-medium tracking-wide text-zinc-600 uppercase rounded-full bg-zinc-100 border border-zinc-200">
          Hello, I'm
        </p>

        <h1 className="mt-6 text-5xl sm:text-6xl md:text-7xl font-bold tracking-tight text-zinc-900">
          范睿峰
        </h1>

        <p className="mt-4 text-xl sm:text-2xl text-zinc-600 leading-relaxed">
          用 AI 构建产品，记录学习与成长
        </p>

        <p className="mt-6 max-w-xl mx-auto text-base text-zinc-500 leading-relaxed">
          用 Python、Next.js 和大模型 API 构建实际能用的产品。这是我的求职主页——你可以直接和我的 AI 助手聊聊我。
        </p>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <a 
            href="#chat" 
            className="inline-flex items-center px-6 py-3 rounded-full bg-zinc-900 text-white font-medium text-base hover:bg-zinc-700 transition-colors"
          >
            和我的 AI 聊聊
            <span className="ml-2">→</span>
          </a>
          <a 
            href="#projects" 
            className="inline-flex items-center px-6 py-3 rounded-full border border-zinc-300 text-zinc-700 font-medium text-base hover:border-zinc-900 hover:text-zinc-900 transition-colors"
          >
            看我的项目
          </a>
        </div>

      </div>
    </section>
  );
}
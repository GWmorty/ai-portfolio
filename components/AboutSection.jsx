export default function AboutSection() {
  return (
    <section id="about" className="py-24 px-6 bg-white border-t border-zinc-100">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <p className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
            About
          </p>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-zinc-900">
            关于我
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="md:col-span-2 space-y-4 text-zinc-700 leading-relaxed">
            <p>
              我是范睿峰，正在转型 AI 工程方向。我的学习方法是「项目驱动」——每学一个概念就做一个项目验证：学完 Python OOP 做了 RAG 原型、学完 Next.js 部署了静态站点、调用大模型 API 做了对话 Demo。
            </p>
            <p>
              这个作品集本身也是一次「边学边做」的实践——前端用 Next.js + Tailwind，未来会接入 FastAPI 后端和 RAG，让你能直接和我的 AI 助手对话。访客体验产品的过程，就是了解我的过程。
            </p>
            <p>
              长期方向是 AI Infrastructure / MLOps，希望能讲清从模型调用到生产部署的完整链路。目前也在探索 AI 项目协助、Agent 开发等更贴近应用的岗位机会。
            </p>
          </div>

          <div className="space-y-3">
            <div className="p-5 rounded-2xl border border-zinc-200 bg-zinc-50">
              <p className="text-xs uppercase tracking-wider text-zinc-500">当前状态</p>
              <p className="mt-1.5 font-medium text-zinc-900">主动求职中</p>
              <p className="text-sm text-zinc-600">可立即入职</p>
            </div>
            <div className="p-5 rounded-2xl border border-zinc-200 bg-zinc-50">
              <p className="text-xs uppercase tracking-wider text-zinc-500">目标方向</p>
              <p className="mt-1.5 font-medium text-zinc-900">AI 应用 / 项目协助</p>
              <p className="text-sm text-zinc-600">长期：AI Infra / MLOps</p>
            </div>
            <div className="p-5 rounded-2xl border border-zinc-200 bg-zinc-50">
              <p className="text-xs uppercase tracking-wider text-zinc-500">在学</p>
              <p className="mt-1.5 font-medium text-zinc-900">零到全栈 · 模块 5</p>
              <p className="text-sm text-zinc-600">FastAPI 后端开发</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
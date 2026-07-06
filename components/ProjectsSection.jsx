export default function ProjectsSection() {
  const projects = [
    {
      title: "AI 求职作品集",
      tag: "进行中",
      description: "你正在浏览的这个网站。让访客通过 AI 对话了解候选人——前端 Next.js + Tailwind，后端即将接入 FastAPI + RAG。访客体验产品的过程，就是了解我的过程。",
      tech: ["Next.js", "Tailwind", "FastAPI", "RAG"],
      link: "#",
      linkLabel: "就在这里",
    },
    {
      title: "miniRGA · RAG 原型",
      tag: "可运行",
      description: "一个轻量级 RAG 系统，用 Jaccard 字符相似度做检索，DeepSeek API 做流式生成。是学习 RAG 完整链路的第一步——下一步会换成向量检索。",
      tech: ["Python", "DeepSeek API", "Jaccard"],
      link: "#",
      linkLabel: "源码即将开源",
    },
    {
      title: "zero-to-tech · 全栈课程项目",
      tag: "已上线",
      description: "李勃老师「零到全栈」课程的项目实战。Next.js 静态导出 + Nginx 部署到云服务器，完整走过从开发到上线的流程。",
      tech: ["Next.js", "Nginx", "Linux"],
      link: "#",
      linkLabel: "在线 demo 即将贴",
    },
  ];

  const tagStyles = {
    "进行中": "bg-amber-100 text-amber-800 border border-amber-200",
    "可运行": "bg-emerald-100 text-emerald-800 border border-emerald-200",
    "已上线": "bg-blue-100 text-blue-800 border border-blue-200",
  };

  return (
    <section id="projects" className="py-24 px-6 bg-white border-t border-zinc-100">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <p className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
            Projects
          </p>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-zinc-900">
            项目经历
          </h2>
          <p className="mt-3 text-zinc-600">
            三个能展示我当前能力的学习项目——每个都是边学边做的产物。
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {projects.map((project) => (
            <article
              key={project.title}
              className="flex flex-col p-6 rounded-2xl border border-zinc-200 bg-zinc-50 hover:border-zinc-300 hover:bg-white hover:shadow-md transition-all"
            >
              <span className={`self-start px-2.5 py-0.5 text-xs font-medium rounded-full ${tagStyles[project.tag]}`}>
                {project.tag}
              </span>
              <h3 className="mt-3 text-lg font-semibold text-zinc-900">
                {project.title}
              </h3>
              <p className="mt-2 flex-1 text-sm text-zinc-600 leading-relaxed">
                {project.description}
              </p>
              <div className="mt-4 flex flex-wrap gap-1.5">
                {project.tech.map((tech) => (
                  <span
                    key={tech}
                    className="px-2 py-0.5 text-xs rounded bg-zinc-200/70 text-zinc-700"
                  >
                    {tech}
                  </span>
                ))}
              </div>
              <a
                href={project.link}
                className="mt-5 inline-flex items-center text-sm font-medium text-zinc-900 hover:underline"
              >
                {project.linkLabel}
                <span className="ml-1">→</span>
              </a>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
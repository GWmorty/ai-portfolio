export default function SkillsSection() {
  const skills = [
    { name: "Python", level: "实战" },
    { name: "Next.js / React", level: "实战" },
    { name: "Tailwind CSS", level: "实战" },
    { name: "FastAPI", level: "实战" },
    { name: "大模型 API", level: "实战" },
    { name: "RAG", level: "实战" },
    { name: "LangGraph / Agent", level: "实战" },
    { name: "Docker", level: "实战" },
    { name: "Linux 部署", level: "实战" },
    { name: "Chroma", level: "学习中" },
    { name: "流式 SSE", level: "学习中" },
    { name: "Eval / 可观测性", level: "学习中" },
  ];

  const levelStyles = {
    实战: "bg-zinc-900 text-white",
    学习中: "bg-zinc-200 text-zinc-800",
    概念: "bg-zinc-100 text-zinc-500 border border-zinc-200",
  };

  return (
    <section id="skills" className="py-24 px-6 bg-zinc-50 border-t border-zinc-100">
      <div className="max-w-5xl mx-auto">
        <div className="mb-10">
          <p className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
            Skills
          </p>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-zinc-900">
            技能栈
          </h2>
          <p className="mt-3 text-zinc-600">
            三级分类——<span className="font-medium text-zinc-900">实战</span>（做过项目）/ <span className="font-medium text-zinc-900">学习中</span>（在写代码）/ <span className="font-medium text-zinc-900">概念</span>（读过笔记未实操）
          </p>
        </div>

        <div className="flex flex-wrap gap-3">
          {skills.map((skill) => (
            <span
              key={skill.name}
              className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${levelStyles[skill.level]}`}
            >
              {skill.name}
              <span className="ml-2 text-xs opacity-70">· {skill.level}</span>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
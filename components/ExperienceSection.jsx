export default function ExperienceSection() {
  const experiences = [
    {
      company: "上海屹力教育科技",
      role: "项目专员（产研部）",
      period: "2024.10 - 2025.01",
      summary: "智慧校园系统项目的全流程交付与运维，练成项目管理和文档体系能力——这些能力可直接迁移到 AI 项目的推进与落地。",
      highlights: [
        "项目全流程：经历客户项目从需求对接、研发跟进到系统验收全过程，平均较计划提前 3 天交付",
        "文档体系：编写《部署指南》《运维手册》，缩短部署周期、提升缺陷修复时效——AI 项目同样需要把复杂技术沉淀成可复用的文档",
        "客户响应：建立 7×12 小时响应机制，平均处理时效 < 4 小时，客户投诉率下降约 50%",
        "跨部门协调：协调开发、产品、客户多方推进，完成 30+ 项功能迭代",
      ],
      tags: ["项目管理全流程", "文档体系", "跨部门协调", "客户响应"],
    },
    {
      company: "临腾数字科技",
      role: "总经理助理",
      period: "2026.04 - 2026.06",
      summary: "在合资子公司梳理多条科技业务线、制作面向客户的商业材料——快速理解陌生业务、把复杂信息结构化表达的能力，正是 AI 项目协调所需要的。",
      highlights: [
        "多业务线快速学习：2 个月内系统掌握智慧园区、云测试、ISP 三条业务线的逻辑与商业模式",
        "结构化表达：独立梳理各业务板块核心内容，制作面向客户的业务介绍材料",
        "行业认知：接触算力租赁等 AI 相关新兴业务方向，对 AI 基础设施的应用场景有第一手认识",
      ],
      tags: ["快速学习", "结构化思维", "商业材料", "AI 业务认知"],
    },
  ];

  return (
    <section id="experience" className="py-24 px-6 bg-zinc-50 border-t border-zinc-100">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12">
          <p className="text-sm font-medium text-zinc-500 uppercase tracking-wider">
            Experience
          </p>
          <h2 className="mt-2 text-3xl sm:text-4xl font-bold text-zinc-900">
            工作经历
          </h2>
          <p className="mt-3 text-zinc-600">
            两段真实工作经历积累的项目管理、文档体系、跨部门协调能力——正是 AI 项目协助岗位需要的底层能力。
          </p>
        </div>

        <div className="space-y-6">
          {experiences.map((exp) => (
            <article
              key={exp.company}
              className="p-6 rounded-2xl border border-zinc-200 bg-white hover:border-zinc-300 hover:shadow-sm transition-all"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div>
                  <h3 className="text-lg font-semibold text-zinc-900">
                    {exp.company}
                  </h3>
                  <p className="text-sm text-zinc-600">{exp.role}</p>
                </div>
                <span className="text-xs font-medium text-zinc-500 px-2.5 py-0.5 rounded-full bg-zinc-100">
                  {exp.period}
                </span>
              </div>

              <p className="mt-3 text-sm text-zinc-700 leading-relaxed">
                {exp.summary}
              </p>

              <ul className="mt-3 space-y-1.5">
                {exp.highlights.map((h, i) => (
                  <li key={i} className="text-sm text-zinc-600 flex gap-2">
                    <span className="text-zinc-400 mt-0.5">·</span>
                    <span>{h}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-4 flex flex-wrap gap-1.5">
                {exp.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-0.5 text-xs rounded bg-zinc-200/70 text-zinc-700"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

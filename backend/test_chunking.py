# test_chunking.py — mini_rag 切块逻辑的单元测试（纯函数，不调任何 API）
#
# 运行：backend 目录下 `pytest test_chunking.py -v`
# CI 部署前会自动跑一遍全部测试（见 .github/workflows/deploy.yml）

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_rag import chunk_file, chunk_directory


def test_h2_splits_into_sections(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n## 技能\nPython 熟练\n## 项目\n做了 RAG\n", encoding="utf-8")
    chunks = chunk_file(str(p))
    assert len(chunks) == 2
    assert chunks[0]["section"] == "技能"
    assert chunks[1]["section"] == "项目"
    assert "Python 熟练" in chunks[0]["text"]


def test_h1_context_prepended(tmp_path):
    p = tmp_path / "b.md"
    p.write_text("# 范睿峰简介\n## 基本信息\n上海\n", encoding="utf-8")
    chunks = chunk_file(str(p))
    # H2 段落会自动带上文件 H1 标题做上下文（检索时知道这段讲谁）
    assert chunks[0]["text"].startswith("# 范睿峰简介")
    assert "上海" in chunks[0]["text"]


def test_no_h2_whole_file_one_chunk(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("没有标题的纯文本\n" * 5, encoding="utf-8")
    chunks = chunk_file(str(p))
    assert len(chunks) == 1


def test_long_section_falls_back_to_char_split(tmp_path):
    p = tmp_path / "d.md"
    body = "字" * (300 * 5)  # 5 倍 chunk_size，超过 4 倍阈值触发字符兜底切分
    p.write_text(f"# 长文\n## 长段\n{body}\n", encoding="utf-8")
    chunks = chunk_file(str(p))
    assert len(chunks) > 1  # 长段被兜底切成多块


def test_chunk_directory_only_md_txt(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## S\n内容 A\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("# B\n## S\n内容 B\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print(1)", encoding="utf-8")  # 应被忽略
    chunks = chunk_directory(str(tmp_path))
    assert {c["source"] for c in chunks} == {"a.md", "b.txt"}


def test_chunk_ids_increment(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## S1\nx\n## S2\ny\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n## S\nz\n", encoding="utf-8")
    chunks = chunk_directory(str(tmp_path))
    ids = [c["id"] for c in chunks]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)

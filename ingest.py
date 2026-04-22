import os
import re
import json
import warnings
import multiprocessing
from pathlib import Path
from markdownify import markdownify as md
import fitz
import pymupdf4llm

RAW_DIR = Path("raw_notes")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

MAX_WORKERS = min(os.cpu_count(), 8)


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    prev_empty = False
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
            continue
        prev_empty = False
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def save_md(filename: str, content: str):
    path = DOCS_DIR / f"{filename}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ingest_txt(src: Path):
    text = src.read_text(encoding="utf-8")
    content = f"# {src.stem}\n\n{clean_text(text)}"
    save_md(src.stem, content)
    print(f"[OK] {src.name}")


def ingest_md(src: Path):
    text = src.read_text(encoding="utf-8")
    if not text.strip().startswith("#"):
        text = f"# {src.stem}\n\n{text}"
    save_md(src.stem, clean_text(text))
    print(f"[OK] {src.name}")


def ingest_html(src: Path):
    html = src.read_text(encoding="utf-8")
    text = md(html, heading_style="ATX")
    if not text.strip().startswith("#"):
        text = f"# {src.stem}\n\n{text}"
    save_md(src.stem, clean_text(text))
    print(f"[OK] {src.name}")


def is_text_pdf(doc: fitz.Document, sample_pages: int = 2) -> bool:
    total_text_len = 0
    for page_num in range(min(sample_pages, len(doc))):
        text = doc[page_num].get_text().strip()
        total_text_len += len(text)
        if len(text) > 200:
            return True
    return (total_text_len / sample_pages) > 100


def extract_text_pdf(doc: fitz.Document, filename: str) -> str:
    """纯文本 PDF 提取，保留基本结构"""
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        page_lines = []
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            text = text.strip()
            if not text or len(text) < 2:
                continue
            page_height = page.rect.height
            # 过滤页眉页脚
            if y0 < 60 or y1 > page_height - 40:
                if len(text) < 30:
                    continue
            # 检测标题
            is_heading = y0 < 150 and len(text) < 50 and not text.endswith('。') and not text.endswith('.')
            if is_heading and not text.startswith('#'):
                page_lines.append(f"## {text}")
            else:
                page_lines.append(text)
        all_text.extend(page_lines)
    
    raw = "\n\n".join(all_text)
    return post_process_raw_text(raw, filename)


def post_process_raw_text(text: str, filename: str) -> str:
    """清洗纯文本提取结果"""
    lines = text.split("\n")
    filtered = []
    seen_headings = set()
    prev_stripped = ""
    skip_keywords = ["内部资料", "请勿外传", "duckcdu", "duckedu", "五道口一只鸭"]
    
    for line in lines:
        stripped = line.strip()
        if stripped == filename and not stripped.startswith("#"):
            continue
        if stripped.startswith("#"):
            heading_text = re.sub(r'^#+\s*', '', stripped)
            if heading_text in seen_headings:
                continue
            seen_headings.add(heading_text)
        if stripped in ['"', '“', '”', '‘', '’', '•', '-', '*', '·', '◦', '▪', '□', '○']:
            continue
        if stripped == prev_stripped:
            continue
        if any(k in stripped for k in skip_keywords) and len(stripped) < 50:
            continue
        filtered.append(line)
        if stripped:
            prev_stripped = stripped
    
    result = [f"# {filename}", ""]
    result.extend(filtered)
    return "\n".join(result)


def analyze_page_header_footer(doc: fitz.Document, sample_pages: int = 3) -> tuple:
    page_heights = []
    text_blocks_by_y = {}
    for page_num in range(min(sample_pages, len(doc))):
        page = doc[page_num]
        page_heights.append(page.rect.height)
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if not text or len(text) < 3:
                continue
            y_bucket = int(y0 / 10) * 10
            if y_bucket not in text_blocks_by_y:
                text_blocks_by_y[y_bucket] = {"count": 0, "texts": set()}
            text_blocks_by_y[y_bucket]["count"] += 1
            text_blocks_by_y[y_bucket]["texts"].add(text[:50])
    
    if not page_heights:
        return (80, 750)
    
    avg_height = sum(page_heights) / len(page_heights)
    header_y = 80
    footer_y = avg_height - 80
    
    for y_bucket, info in text_blocks_by_y.items():
        if info["count"] >= sample_pages * 0.8:
            y_pos = y_bucket
            if y_pos < avg_height * 0.15:
                header_y = max(header_y, y_pos + 20)
            elif y_pos > avg_height * 0.85:
                footer_y = min(footer_y, y_pos - 10)
    
    return (header_y, footer_y)


def is_likely_header_footer(text: str, y: float, header_y: float, footer_y: float) -> bool:
    if y < header_y or y > footer_y:
        return True
    if len(text.strip()) < 20:
        patterns = [r'^\d+$', r'^\d+/\d+$', r'^[第]?\d+[页]$', r'^\w+\.(com|cn|edu|org)']
        for p in patterns:
            if re.search(p, text.strip()):
                return True
    return False


def post_process_md(md_text: str, filename: str, valid_texts: list = None) -> str:
    """OCR PDF 的 Markdown 后处理"""
    lines = md_text.split("\n")
    n = len(lines)
    
    toc_lines = []
    content_lines = []
    i = 0
    
    while i < n:
        line = lines[i]
        stripped = line.strip()
        
        if stripped.isdigit() and len(stripped) <= 3:
            i += 1
            continue
        
        # 检测目录：包含"目录"关键词的行 + 后续表格
        if stripped in ["# 目录", "## 目录", "### 目录", "目录"]:
            toc_lines.append(line)
            i += 1
            while i < n:
                curr = lines[i].strip()
                if curr.startswith("|") or not curr:
                    toc_lines.append(lines[i])
                    i += 1
                else:
                    break
            continue
        
        # 检测孤立表格行（可能是目录的一部分）
        if stripped.startswith("|") and re.search(r'\|\s*\*?\d+\*?\s*\|', stripped):
            is_orphan_toc = False
            for j in range(max(0, i - 10), i):
                if "目录" in lines[j] or (lines[j].strip().startswith("|") and j > 0 and lines[j-1].strip().startswith("|")):
                    is_orphan_toc = True
                    break
            if is_orphan_toc:
                while i < n:
                    curr = lines[i].strip()
                    if curr.startswith("|") or not curr:
                        toc_lines.append(lines[i])
                        i += 1
                    else:
                        break
                continue
        
        content_lines.append(line)
        i += 1
    
    filtered = []
    seen_headings = set()
    prev_stripped = ""
    
    for line in content_lines:
        stripped = line.strip()
        
        if stripped == filename and not stripped.startswith("#"):
            continue
        
        if stripped.startswith("#"):
            heading_text = re.sub(r'^#+\s*', '', stripped)
            if heading_text in seen_headings:
                continue
            seen_headings.add(heading_text)
        
        if stripped in ['"', '“', '”', '‘', '’', '•', '-', '*', '·', '◦', '▪', '□', '○']:
            continue
        
        if re.match(r'^\|\s*\|', stripped):
            continue
        
        if re.match(r'^[\-\|=\.]{3,}$', stripped):
            continue
        
        if stripped == prev_stripped:
            continue
        
        if valid_texts is not None and stripped and len(stripped) > 30:
            is_valid = False
            for vt in valid_texts:
                if stripped[:30] in vt or vt[:30] in stripped:
                    is_valid = True
                    break
            if not is_valid:
                continue
        
        filtered.append(line)
        if stripped:
            prev_stripped = stripped
    
    result = [f"# {filename}", ""]
    
    # 添加折叠目录（如果有）
    if toc_lines:
        clean_toc = []
        for line in toc_lines:
            stripped = line.strip()
            if not stripped:
                clean_toc.append(line)
                continue
            if re.match(r'^\|\s*\*?\d+\*?\s*\|$', stripped):
                continue
            if re.match(r'^\|[-\s|]+\|$', stripped):
                continue
            if stripped in ["# 目录", "## 目录", "### 目录", "目录"]:
                continue
            clean_toc.append(line)
        
        toc_content = "\n".join(clean_toc).strip()
        if toc_content:
            result.extend([
                "<details>",
                "<summary>📑 目录（点击展开）</summary>",
                "",
                toc_content,
                "",
                "</details>",
                ""
            ])
    
    result.extend(filtered)
    return "\n".join(result)


def process_single_pdf(src_path: Path) -> dict:
    result = {"path": str(src_path), "success": False, "error": None, "method": None}
    
    try:
        doc = fitz.open(src_path)
        
        if is_text_pdf(doc):
            result["method"] = "fast_text"
            md_content = extract_text_pdf(doc, src_path.stem)
        else:
            result["method"] = "ocr"
            warnings.filterwarnings("ignore")
            md_text = pymupdf4llm.to_markdown(
                src_path,
                pages=None,
                write_images=False,
                image_path=None,
            )
            header_y, footer_y = analyze_page_header_footer(doc)
            valid_texts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("blocks")
                for block in blocks:
                    x0, y0, x1, y1, text, *_ = block
                    text = text.strip()
                    if not text:
                        continue
                    if is_likely_header_footer(text, y0, header_y, footer_y):
                        continue
                    valid_texts.append(text)
            
            md_content = post_process_md(md_text, src_path.stem, valid_texts)
        
        doc.close()
        content = clean_text(md_content)
        save_md(src_path.stem, content)
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def ingest_ipynb(src: Path):
    data = json.loads(src.read_text(encoding="utf-8"))
    chunks = []
    for cell in data.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown" and source.strip():
            chunks.append(source)
        elif cell_type == "code" and source.strip():
            chunks.append(f"```python\n{source}\n```")
    content = f"# {src.stem}\n\n" + "\n\n".join(chunks)
    save_md(src.stem, content)
    print(f"[OK] {src.name}")


def main():
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print(f"!!! 请先把原始笔记放进 {RAW_DIR}/ 目录")
        return

    pdf_files = []
    other_tasks = []
    
    for src in RAW_DIR.iterdir():
        suffix = src.suffix.lower()
        if suffix == ".pdf":
            pdf_files.append(src)
        elif suffix in [".md", ".txt", ".html", ".ipynb"]:
            other_tasks.append((suffix, src))
    
    for suffix, src in other_tasks:
        if suffix == ".md":
            ingest_md(src)
        elif suffix == ".txt":
            ingest_txt(src)
        elif suffix == ".html":
            ingest_html(src)
        elif suffix == ".ipynb":
            ingest_ipynb(src)
    
    if pdf_files:
        print(f"\n[并行处理] {len(pdf_files)} 个 PDF，使用 {MAX_WORKERS} 进程...")
        
        with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
            results = pool.map(process_single_pdf, pdf_files)
        
        fast_count = sum(1 for r in results if r["success"] and r["method"] == "fast_text")
        ocr_count = sum(1 for r in results if r["success"] and r["method"] == "ocr")
        fail_count = sum(1 for r in results if not r["success"])
        
        print(f"\n[统计] 纯文本快速提取: {fast_count}, OCR 提取: {ocr_count}, 失败: {fail_count}")
        for r in results:
            if not r["success"]:
                print(f"  ✗ {r['path']}: {r['error']}")

    print(f"\n全部完成。输出目录：{DOCS_DIR}/")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
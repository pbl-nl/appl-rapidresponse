"""Step 2: turn every source file into passages with metadata.

PDFs are read page by page. Empty or unreadable pages get OCR with PaddleOCR.
Excel tables give one passage per table row.
Results go to outputs/02_text_processing/.

Usage:
    python src/02_text_processing.py --file Verpact   # test on one file, saved in a test folder
    python src/02_text_processing.py                  # all group A and B files
    python src/02_text_processing.py --summary-only   # only remake the summary
"""
import argparse
import collections
import csv
import json
import importlib.util
import re
import sys
import unicodedata
from pathlib import Path

import openpyxl
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from text_utils import join_lines  # noqa: E402

CITATION_MAP = config.STEP1_DIR / "citation_map.csv"
TEST_DIR = config.STEP2_DIR / "test"  # a run with --file goes here, so the full output stays as it is
PASSAGES_NAME = "passages.jsonl"
PAGE_REPORT_NAME = "page_report.csv"
SUMMARY_NAME = "text_processing_summary.txt"

HAS_OCR = importlib.util.find_spec("paddleocr") is not None
MIN_PAGE_CHARS = 50      # fewer characters than this means the page is empty
MIN_GOOD_WORDS = 0.5     # share of normal looking words below this means garbage
MIN_PROSE_LINE = 12      # OCR: a line at least 12 times as wide as it is high is running text
HEADER_ZONE = 0.08       # top and bottom 8% of the page can hold headers and footers
WORD_RE = re.compile(r"[(\[‘'\"“]*[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9’'\-./,%€]*[)\]’'\"”.,;:!?%]*")


# Files
def usable_files():
    """{file name: citation} for the files of group A and B. One file can have two citations."""
    files = collections.defaultdict(list)
    with open(CITATION_MAP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["group"] in ("A", "B") and r["file"]:
                files[r["file"]].append(r["citation"])
    return {name: "; ".join(c) for name, c in files.items()}


# Page check and OCR
def good_word_share(text):
    """Share of words that look like normal words or numbers. Garbage text scores low."""
    words = text.split()
    if not words:
        return 0.0
    good = [w for w in words if WORD_RE.fullmatch(w) and re.search(r"[aeiouyAEIOUY0-9]", w)]
    return len(good) / len(words)


def page_status(page):
    text = page.get_text()
    if len(text.strip()) < MIN_PAGE_CHARS:
        return "empty"
    if good_word_share(text) < MIN_GOOD_WORDS:
        return "garbage"
    return "ok"


def column_gap(page):
    """x position (0 to 1) of an empty vertical strip between two text columns, or None.

    The page is rendered in grey. Near the middle of the page (42% to 58% of the width)
    we look for a strip of pixel columns without any ink between the header and footer bands.
    Two-column layouts have their gap there. A glossary with a narrow left column is not split.
    """
    pix = page.get_pixmap(dpi=100, colorspace=pymupdf.csGRAY)
    w, h, data = pix.width, pix.height, pix.samples
    y0, y1 = int(h * HEADER_ZONE), int(h * (1 - HEADER_ZONE))
    x0 = int(w * 0.42)
    empty = []
    for x in range(x0, int(w * 0.58)):
        ink = any(data[y * pix.stride + x] < 200 for y in range(y0, y1))
        empty.append(not ink)
    # Longest run of empty pixel columns
    best, run, best_end = 0, 0, 0
    for i, e in enumerate(empty):
        run = run + 1 if e else 0
        if run > best:
            best, best_end = run, i
    if best < 4:  # 4 pixels at 100 dpi is about 1 mm
        return None
    return (x0 + best_end - best / 2) / w


_ocr_model = None


def ocr_model():
    """The PaddleOCR model, loaded once on first use (loading takes a few seconds)."""
    global _ocr_model
    if _ocr_model is None:
        from paddleocr import PaddleOCR
        _ocr_model = PaddleOCR(
            text_detection_model_name=f"{config.OCR_MODEL}_det",
            text_recognition_model_name=f"{config.OCR_MODEL}_rec",
            use_doc_orientation_classify=False,  # pages are already upright
            use_doc_unwarping=False,             # rendered PDF pages are flat
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _ocr_model


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def ocr_lines(image):
    """Text lines of an image as dicts with text, x0, top, x1, bottom in pixels.

    PaddleOCR can return one line as several boxes. Boxes are joined into one line when
    they overlap vertically and the space between them is small. Boxes far apart on the
    same height (two columns, table cells) stay separate lines.
    """
    boxes = []
    for res in ocr_model().predict(image):
        for text, (x0, y0, x1, y1) in zip(res["rec_texts"], res["rec_boxes"]):
            if text.strip():
                boxes.append({"text": text.strip(), "x0": int(x0), "top": int(y0), "x1": int(x1), "bottom": int(y1)})
    lines = []
    for b in sorted(boxes, key=lambda b: b["x0"]):
        h = b["bottom"] - b["top"]
        for line in lines:
            overlap = min(line["bottom"], b["bottom"]) - max(line["top"], b["top"])
            if overlap > 0.5 * min(h, line["bottom"] - line["top"]) and -h < b["x0"] - line["x1"] < 1.5 * h:
                line["text"] += " " + b["text"]
                line["x1"] = max(line["x1"], b["x1"])
                line["top"], line["bottom"] = min(line["top"], b["top"]), max(line["bottom"], b["bottom"])
                break
        else:
            lines.append(dict(b))
    return lines


def ocr_paragraphs(lines):
    """Join lines into paragraphs. A line continues a paragraph when it starts just below
    it (less than one line height of empty space), overlaps it horizontally, and has
    about the same height (so a heading is not joined with body text).

    The line before must also be long, like a line of running text. A short line ends
    a paragraph, and short table cells are never joined, so a table is read row by row.
    """
    paragraphs = []
    for line in sorted(lines, key=lambda l: l["top"]):
        h = line["bottom"] - line["top"]
        best = None
        for par in paragraphs:
            par_h = median(par["heights"])
            gap = line["top"] - par["bottom"]
            if (-0.3 * par_h < gap < 1.0 * par_h
                    and min(line["x1"], par["x1"]) - max(line["x0"], par["x0"]) > 0
                    and 0.7 < h / par_h < 1.4
                    and par["last_width"] >= MIN_PROSE_LINE * par_h
                    and (best is None or gap < best[0])):
                best = (gap, par)
        if best:
            par = best[1]
            par["lines"].append(line["text"])
            par["heights"].append(h)
            par["x0"], par["x1"] = min(par["x0"], line["x0"]), max(par["x1"], line["x1"])
            par["bottom"] = max(par["bottom"], line["bottom"])
            par["last_width"] = line["x1"] - line["x0"]
        else:
            paragraphs.append({"lines": [line["text"]], "heights": [h],
                               "x0": line["x0"], "x1": line["x1"], "top": line["top"], "bottom": line["bottom"],
                               "last_width": line["x1"] - line["x0"]})
    return reading_order(paragraphs)


def reading_order(items):
    """Sort boxes in reading order with a recursive XY-cut: split at empty horizontal
    bands (top to bottom), otherwise at empty vertical bands (left to right, so columns
    are read one after the other)."""
    if len(items) <= 1:
        return items
    for start, end in (("top", "bottom"), ("x0", "x1")):
        items = sorted(items, key=lambda b: b[start])
        groups, group, reach = [], [items[0]], items[0][end]
        for b in items[1:]:
            if b[start] >= reach:
                groups.append(group)
                group = []
            group.append(b)
            reach = max(reach, b[end])
        groups.append(group)
        if len(groups) > 1:
            return [b for g in groups for b in reading_order(g)]
    return sorted(items, key=lambda b: (b["top"], b["x0"]))


def ocr_region(page, rect):
    """OCR one part of a page with PaddleOCR. Returns blocks like text_blocks()."""
    import numpy as np
    dpi = config.OCR_DPI
    pix = page.get_pixmap(dpi=dpi, clip=rect, colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    image = np.ascontiguousarray(image[:, :, ::-1])  # RGB to BGR, which PaddleOCR expects
    paragraphs = ocr_paragraphs(ocr_lines(image))

    blocks = []
    scale = 72 / dpi  # pixels to PDF points
    for par in paragraphs:
        lines = [unicodedata.normalize("NFKC", line) for line in par["lines"]]
        text = lines[0]
        for line in lines[1:]:
            text = join_lines(text, line)
        blocks.append({
            "text": text,
            "size": median(par["heights"]) * scale,
            "bold": False,
            "top": (rect.y0 + par["top"] * scale) / page.rect.height,
            "bottom": (rect.y0 + par["bottom"] * scale) / page.rect.height,
        })
    return blocks


def ocr_blocks(page):
    """Text blocks of a page read with OCR.

    When the page has two columns, the left column, the right column and the header
    and footer bands are read separately, so lines of the two columns do not mix.
    """
    r = page.rect
    gap = column_gap(page)
    if gap is None:
        return ocr_region(page, r)
    top, bottom = r.y0 + r.height * HEADER_ZONE, r.y1 - r.height * HEADER_ZONE
    middle = r.x0 + r.width * gap
    regions = [
        pymupdf.Rect(r.x0, r.y0, r.x1, top),
        pymupdf.Rect(r.x0, top, middle, bottom),
        pymupdf.Rect(middle, top, r.x1, bottom),
        pymupdf.Rect(r.x0, bottom, r.x1, r.y1),
    ]
    return [b for rect in regions for b in ocr_region(page, rect)]


# Blocks and passages
def text_blocks(page):
    """Paragraph blocks of a page as dicts with text, font size, bold, and vertical position.

    Blocks keep the order of the PDF, which follows the columns. Lines inside a block are
    joined. A block that looks like a table (several short lines, most with a number)
    keeps its line breaks, so table rows and their numbers stay together.
    """
    page_dict = page.get_text("dict")
    blocks = []
    for b in page_dict["blocks"]:
        lines = []
        sizes, bold = [], False
        for line in b.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).replace("­", "")  # soft hyphen
            text = unicodedata.normalize("NFKC", text)  # ligatures become normal letters
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                lines.append(text)
                sizes += [s["size"] for s in line["spans"] if s["text"].strip()]
                bold = bold or any("Bold" in s["font"] or s["flags"] & 16 for s in line["spans"])
        if not lines:
            continue
        with_number = sum(bool(re.search(r"\d", l)) for l in lines)
        is_table = len(lines) >= 3 and sum(map(len, lines)) / len(lines) < 35 and with_number >= len(lines) / 2
        if is_table:
            text = "\n".join(lines)
        else:
            text = lines[0]
            for line in lines[1:]:
                text = join_lines(text, line)
        blocks.append({
            "text": text,
            "size": max(sizes),
            "bold": bold,
            "top": b["bbox"][1] / page.rect.height,
            "bottom": b["bbox"][3] / page.rect.height,
        })
    return blocks


def repeated_lines(pages_blocks):
    """Texts that sit at the top or bottom of at least 30% of the pages: headers and footers.

    Digits are replaced by '#' first, so 'PBL | 12' and 'PBL | 13' count as the same text.
    """
    counts = collections.Counter()
    for blocks in pages_blocks:
        seen = {re.sub(r"\d+", "#", b["text"]) for b in blocks
                if b["top"] < HEADER_ZONE or b["bottom"] > 1 - HEADER_ZONE}
        counts.update(seen)
    limit = max(3, 0.3 * len(pages_blocks))
    return {t for t, n in counts.items() if n >= limit}


def is_heading(block, body_size):
    """A heading is clearly bigger than body text, or bold, short and without a full stop."""
    text = block["text"]
    if len(text) > 150 or "\n" in text:
        return False
    return block["size"] >= body_size * 1.2 or (block["bold"] and not text.endswith("."))


def split_long(text, limit):
    """Split a text that is longer than the limit at sentence ends."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    parts, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > limit:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def page_passages(blocks, body_size):
    """Split the blocks of one page into passages of a few paragraphs."""
    passages, current = [], []

    def close():
        if current:
            passages.append("\n\n".join(current))
            current.clear()

    for b in blocks:
        # A sentence that goes on in the next column or block stays in one paragraph
        if current and not re.search(r"[.!?:;)”\"]$", current[-1]) and re.match(r"[a-zà-ÿ]", b["text"]):
            current[-1] = join_lines(current[-1], b["text"])
            continue
        size = sum(map(len, current))
        if is_heading(b, body_size):
            # A heading starts a new passage, unless the current one is still very short
            if size >= config.PASSAGE_MIN_CHARS:
                close()
            current.append(b["text"])
            continue
        if current and size + len(b["text"]) > config.PASSAGE_MAX_CHARS and size >= config.PASSAGE_MIN_CHARS:
            close()
        if len(b["text"]) > config.PASSAGE_MAX_CHARS and "\n" not in b["text"]:
            for part in split_long(b["text"], config.PASSAGE_MAX_CHARS):
                current.append(part)
                close()
        else:
            current.append(b["text"])
    close()
    # A short passage joins the next one, or the one before it at the end of the page
    merged = []
    for text in passages:
        if merged and len(merged[-1]) < config.PASSAGE_MIN_CHARS:
            merged[-1] += "\n\n" + text
        else:
            merged.append(text)
    if len(merged) > 1 and len(merged[-1]) < config.PASSAGE_MIN_CHARS:
        last = merged.pop()
        merged[-1] += "\n\n" + last
    return merged


def process_pdf(path, citation):
    """Return (passages, page rows) for one PDF."""
    doc = pymupdf.open(path)
    statuses, pages_blocks = [], []
    for page in doc:
        status = page_status(page)
        blocks = []
        if status == "ok":
            blocks = text_blocks(page)
        elif HAS_OCR:
            blocks = ocr_blocks(page)
            status = f"ocr ({status})"
        else:
            status = f"needs_ocr ({status})"
        statuses.append(status)
        pages_blocks.append(blocks)

    repeated = repeated_lines(pages_blocks)
    sizes = collections.Counter()
    for blocks in pages_blocks:
        for b in blocks:
            sizes[round(b["size"], 1)] += len(b["text"])
    body_size = sizes.most_common(1)[0][0] if sizes else 10

    passages, page_rows = [], []
    for number, (status, blocks) in enumerate(zip(statuses, pages_blocks), start=1):
        blocks = [b for b in blocks if re.sub(r"\d+", "#", b["text"]) not in repeated]
        texts = page_passages(blocks, body_size)
        page_rows.append({"file": path.name, "page": number, "status": status,
                          "chars": sum(len(t) for t in texts), "passages": len(texts)})
        for text in texts:
            passages.append({"citation": citation, "source_file": path.name, "page": number, "text": text})
    return passages, page_rows


def cell_text(value):
    """A cell value as text. Whole numbers without '.0', dates as yyyy-mm-dd."""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return re.sub(r"\s+", " ", str(value)).strip()


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def process_excel(path, citation):
    """Return (passages, sheet rows) for one Excel file.

    A sheet with a table becomes one passage per table row. Every number is written with
    its column name, so 'Energiedragers: Delfstoffenwinning = 14838'. Zero values are left out.
    A sheet without a table becomes text passages, like a PDF page. 'page' is the sheet name.
    """
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    passages, sheet_rows = [], []
    for ws in wb.worksheets:
        rows = [[c for c in r] for r in ws.iter_rows(values_only=True) if any(c is not None for c in r)]
        # The header row has three or more text cells, and the rows below it hold numbers
        header_i = next((i for i, r in enumerate(rows)
                         if sum(isinstance(c, str) for c in r) >= 3
                         and any(is_number(c) for row in rows[i + 1:i + 4] for c in row)), None)
        texts = []
        if header_i is None:
            paragraphs = [" ".join(cell_text(c) for c in r if c is not None) for r in rows]
            blocks = [{"text": p, "size": 10, "bold": False, "top": 0.5, "bottom": 0.5} for p in paragraphs]
            texts = page_passages(blocks, body_size=10)
        else:
            title = " ".join(cell_text(c) for r in rows[:header_i] for c in r if c is not None)
            header = [cell_text(c) if c is not None else "" for c in rows[header_i]]
            for r in rows[header_i + 1:]:
                label = " ".join(cell_text(c) for c in r if isinstance(c, str))
                values = [f"{header[i]} = {cell_text(c)}" for i, c in enumerate(r)
                          if is_number(c) and c != 0 and i < len(header) and header[i]]
                if label and values:
                    texts.append(f"{title}. {label}: " + "; ".join(values))
        sheet_rows.append({"file": path.name, "page": ws.title, "status": "excel",
                           "chars": sum(len(t) for t in texts), "passages": len(texts)})
        for text in texts:
            passages.append({"citation": citation, "source_file": path.name, "page": ws.title, "text": text})
    return passages, sheet_rows


def write_summary(out_dir):
    """Make text_processing_summary.txt from passages.jsonl and page_report.csv.

    It has the totals, one line per file, and the quality checks. It is also printed.
    """
    passages_file = out_dir / PASSAGES_NAME
    summary_file = out_dir / SUMMARY_NAME
    with open(passages_file, encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]
    with open(out_dir / PAGE_REPORT_NAME, encoding="utf-8") as f:
        pages = list(csv.DictReader(f))

    lengths = sorted(len(p["text"]) for p in passages)
    file_names = list(dict.fromkeys(p["file"] for p in pages))
    n_excel = sum(Path(n).suffix.lower() in (".xlsx", ".xlsm") for n in file_names)
    pdf_pages = [p for p in pages if p["status"] != "excel"]
    garbage_left = [p for p in passages if good_word_share(p["text"]) < 0.6]
    ids = [p["passage_id"] for p in passages]

    lines = [
        f"Files processed: {len(file_names)} ({len(file_names) - n_excel} PDF, {n_excel} Excel)",
        f"PDF pages: {len(pdf_pages)}",
        f"  read directly: {sum(p['status'] == 'ok' for p in pdf_pages)}",
        f"  read with OCR: {sum(p['status'].startswith('ocr') for p in pdf_pages)}"
        f" (empty {sum(p['status'] == 'ocr (empty)' for p in pdf_pages)},"
        f" garbage {sum(p['status'] == 'ocr (garbage)' for p in pdf_pages)})",
        f"  not read, OCR needed: {sum(p['status'].startswith('needs_ocr') for p in pdf_pages)}",
        f"Excel sheets: {len(pages) - len(pdf_pages)}",
        f"Passages: {len(passages)}",
        f"Passage length in characters: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}",
        f"passages.jsonl size: {passages_file.stat().st_size / 1e6:.1f} MB",
        "",
        "Quality checks:",
        f"  passage ids unique: {'yes' if len(set(ids)) == len(ids) else 'NO'}",
        f"  pages without any passage (blank or figure only): {sum(p['passages'] == '0' for p in pages)}",
        f"  passages under 100 characters: {sum(l < 100 for l in lengths)}",
        f"  passages over 2000 characters: {sum(l > 2000 for l in lengths)}",
        f"  passages that still look like garbage: {len(garbage_left)}",
        "",
        "Per file: pages | OCR pages | passages | file",
    ]
    for name in file_names:
        rows = [p for p in pages if p["file"] == name]
        ocr = sum(p["status"].startswith("ocr") for p in rows)
        n_pass = sum(int(p["passages"]) for p in rows)
        kind = "sheets" if rows[0]["status"] == "excel" else "pages"
        lines.append(f"  {len(rows):4} {kind:6} | {ocr:3} OCR | {n_pass:4} passages | {name}")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsummary -> {summary_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="test on the file whose name starts with this text")
    parser.add_argument("--summary-only", action="store_true",
                        help="only make the summary from the existing outputs, without reading the files again")
    args = parser.parse_args()

    out_dir = TEST_DIR if args.file else config.STEP2_DIR
    if args.summary_only:
        write_summary(out_dir)
        return

    # File names to paths, in references/ and all folders inside it
    paths = {p.name: p for p in config.REFERENCES_DIR.rglob("*") if p.is_file()}
    files = usable_files()
    if args.file:
        files = {n: c for n, c in files.items() if n.startswith(args.file)}

    all_passages, all_pages = [], []
    for name, citation in sorted(files.items()):
        path = paths.get(name)
        if path is None:
            print(f"not found in references/: {name}")
            continue
        if path.suffix.lower() in (".xlsx", ".xlsm"):
            passages, pages = process_excel(path, citation)
        else:
            passages, pages = process_pdf(path, citation)
        all_passages += passages
        all_pages += pages
        statuses = collections.Counter(p["status"] for p in pages)
        print(f"{len(pages):4} pages, {len(passages):4} passages | {dict(statuses)} | {name[:60]}")

    for i, p in enumerate(all_passages, start=1):
        p["passage_id"] = f"P{i:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    passages_file = out_dir / PASSAGES_NAME
    page_report = out_dir / PAGE_REPORT_NAME
    with open(passages_file, "w", encoding="utf-8") as f:
        for p in all_passages:
            row = {"passage_id": p["passage_id"], **{k: p[k] for k in ("citation", "source_file", "page", "text")}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(page_report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "page", "status", "chars", "passages"])
        writer.writeheader()
        writer.writerows(all_pages)

    print(f"\n{len(all_passages)} passages -> {passages_file}")
    print(f"page report -> {page_report}")
    if not HAS_OCR:
        print("PaddleOCR is not installed, so empty and garbage pages were skipped (needs_ocr).")
    print()
    write_summary(out_dir)


if __name__ == "__main__":
    main()

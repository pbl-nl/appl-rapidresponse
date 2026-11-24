"""Step 0: find the cited sentences in the real chapter.

Reads the chapter pages from the report PDF, splits the text into sentences,
keeps the section and page of each sentence, and keeps the sentences that
contain a citation. One row per citation. Step 1 uses the citations, step 8 uses the
sentences as the evaluation set.

Output: outputs/00_cited_sentences/cited_sentences.csv with columns page, section, citation, sentence,
and the same rows in outputs/00_cited_sentences/cited_sentences.json.

Usage:
    python src/00_cited_sentences.py            # whole chapter
    python src/00_cited_sentences.py --page 52  # one page, printed only
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from text_utils import join_lines  # noqa: E402

REPORT = pymupdf.open(config.REPORT_PDF)

OUT_FILE = config.STEP0_DIR / "cited_sentences.csv"
OUT_JSON = config.STEP0_DIR / "cited_sentences.json"

# Lines that are not running text
FOOTER_RE = re.compile(r"^PBL \| \d+$")
FIGURE_RE = re.compile(r"^Figuur \d+\.\d+$")
SECTION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\.? +(\S.*)$")  # "3.3.2. Productgroepen ..."
BOX_RE = re.compile(r"^Tekstkader \d+\.\d+:")

# Abbreviations that end with a full stop but do not end a sentence
ABBREVIATIONS = ("et al.", "bijv.", "o.a.", "ca.", "nr.", "e.d.", "zgn.", "d.w.z.", "m.n.", "i.e.", "e.g.", "resp.")

# Citations
YEAR = r"(?:19|20)\d{2}[a-z]?"
BRACKET_RE = re.compile(r"\(([^()]*)\)")
# One part of a bracket, e.g. "Hanemaaijer et al. 2025" or "bron: Van der Have & Van Bellen 2025, in druk"
AUTHOR_YEAR_RE = re.compile(
    rf"^(?:bron: |zie ook |zie )?([A-Z][^;]{{0,80}}?),? ({YEAR})(?:, in druk|, in voorbereiding|, te verschijnen|: \d+(?:-\d+)?)?$"
)
CLO_RE = re.compile(r"\bCLO \d{4}\b")
# Citation inside a sentence, e.g. "Oates & Verveld (2024)"
NAME = r"(?:[A-Z][\w\-’']+|van|der|de|den|het|&|et al\.)"
NARRATIVE_RE = re.compile(rf"([A-Z][\w\-’']+(?: {NAME})*) \(({YEAR})\)")


def line_kind(span, text):
    """Kind of a line, from the font of its first part.

    Section headings use the 'Heading' font, subheadings are size 12 (bold or italic),
    text box titles start with 'Tekstkader', body text is size 10.5.
    """
    if "Heading" in span["font"]:
        return "heading"
    if span["size"] >= 11.5:
        return "subheading"
    if BOX_RE.match(text):
        return "box"
    return "text"


def page_lines(page):
    """(kind, text) lines of one PDF page, read with PyMuPDF.

    A heading that wraps onto two lines becomes one line.
    Footer and figure labels are dropped.
    """
    lines = []
    page_dict = REPORT[page - 1].get_text("dict")
    for block in page_dict["blocks"]:
        for line in block.get("lines", []):
            text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
            if not text or FOOTER_RE.match(text) or FIGURE_RE.match(text):
                continue
            kind = line_kind(line["spans"][0], text)
            # A heading that wraps onto a second line, but a new section number starts a new heading
            wraps = kind in ("heading", "subheading") and lines and lines[-1][0] == kind \
                and not SECTION_RE.match(text)
            if wraps:
                lines[-1] = (kind, join_lines(lines[-1][1], text))
            else:
                lines.append((kind, text))
    return lines


def build_blocks(first, last):
    """Group the lines of the chapter into text blocks.

    A block is a run of text within one section, stored as a list of (page, line).
    Headings, subheadings and box titles end a block, so they never glue onto a sentence.
    Returns a list of (section, [(page, line), ...], kind) where kind is "text" or "heading".
    """
    blocks = []
    section = "In het kort"
    current = []

    def flush():
        nonlocal current
        if current:
            blocks.append((section, current, "text"))
        current = []

    for page in range(first, last + 1):
        for kind, line in page_lines(page):
            if kind == "heading":
                flush()
                m = SECTION_RE.match(line)
                if m:  # "3.2. Grondstoffengebruik en effecten"
                    section = f"{m.group(1)} {m.group(2)}"
                elif line == "In het kort":
                    section = line
                # The chapter title "3. Grondstoffen" is skipped
            elif kind in ("subheading", "box"):
                flush()
                blocks.append((section, [(page, line)], "heading"))
            else:
                current.append((page, line))
    flush()
    return blocks


def split_sentences(text):
    """Split a block into sentences. Never splits inside brackets or after an abbreviation.

    Returns (start offset, sentence) pairs.
    """
    sentences, start, depth = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch in ".?!" and depth == 0:
            rest = text[i + 1:]
            if not rest.startswith(" ") or len(rest) < 2:
                continue
            nxt = rest[1]
            if not (nxt.isupper() or nxt.isdigit() or nxt in "‘'\"“("):
                continue
            if text[: i + 1].endswith(ABBREVIATIONS):
                continue
            sentences.append((start, text[start: i + 1]))
            start = i + 1
    if text[start:].strip():
        sentences.append((start, text[start:]))
    # Move each offset past leading spaces, so it points at the first word
    return [(s + len(t) - len(t.lstrip()), t.strip()) for s, t in sentences]


def find_citations(sentence):
    """All citations in a sentence, in order, without duplicates."""
    found = []
    for m in BRACKET_RE.finditer(sentence):
        inner = m.group(1)
        for part in inner.split(";"):
            part = part.strip()
            am = AUTHOR_YEAR_RE.match(part)
            if am and len(am.group(1).split()) <= 10:
                found.append((m.start(), f"{am.group(1)} {am.group(2)}"))
    for m in CLO_RE.finditer(sentence):
        found.append((m.start(), m.group(0)))
    for m in NARRATIVE_RE.finditer(sentence):
        name = re.sub(r"^(?:Zie|Volgens|Ook|En) ", "", m.group(1))
        found.append((m.start(), f"{name} {m.group(2)}"))
    found.sort()
    citations = []
    for _, c in found:
        if c not in citations:
            citations.append(c)
    return citations


def cited_rows(first, last):
    rows = []
    for section, lines, kind in build_blocks(first, last):
        # Join the lines and remember at which offset each page starts
        text, page_starts = "", []
        for page, line in lines:
            text = join_lines(text, line) if text else line
            page_starts.append((len(text) - len(line), page))
        sentences = split_sentences(text) if kind == "text" else [(0, text)]
        for offset, sentence in sentences:
            page = [p for start, p in page_starts if start <= offset][-1]
            for citation in find_citations(sentence):
                rows.append({"page": page, "section": section, "citation": citation, "sentence": sentence})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, help="test on one page, print only")
    args = parser.parse_args()

    if args.page:
        for r in cited_rows(args.page, args.page):
            print(f"[{r['page']}] {r['section']} | {r['citation']}\n    {r['sentence']}\n")
        return

    first, last = config.CHAPTER_PAGES
    rows = cited_rows(first, last)
    config.STEP0_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "section", "citation", "sentence"])
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"{len(rows)} rows, {len({r['sentence'] for r in rows})} sentences -> {OUT_FILE}, {OUT_JSON}")


if __name__ == "__main__":
    main()

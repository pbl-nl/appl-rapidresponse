"""Step 1: link every citation in the chapter to a source file.

Reads the citations from cited_sentences.csv and the file names in references/,
then gives each citation a group (A, B, C or no_file) with a note.
Results go to outputs/01_collect_sources/.

Usage:
    python src/01_citation_map.py
"""
import csv
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CITED_FILE = config.STEP0_DIR / "cited_sentences.csv"
OUT_FILE = config.STEP1_DIR / "citation_map.csv"
UNUSED_FILE = config.STEP1_DIR / "unused_files.csv"
SUMMARY_FILE = config.STEP1_DIR / "citation_map_summary.txt"

# Citations whose file name points to the wrong file or is incomplete.
# These are linked to the file that starts with the given name, with the given group.
CORRECTIONS = {
    # citation: (file name starts with, group, note)
    "CBS 2024": ("CBS 2024b", "A", "the sentence (15 percent drop, 1996 to 2022) "
                                   "fits CBS 2024b, not the CBS 2024a file name"),
    "Copper8 2024": ("Copper8 & Circular Finance Lab 2024", "A",
                     "The reference in the chapter is incomplete, same document"),
}
# Citations with a file that does not hold what the chapter cites (group C, skipped)
CONFLICTS = {
    "RVO 2024": "The RVO website is always current (now 2026), "
                "the situation in 2024 cannot be found in this file",
}
# Citations whose file is known to be missing
KNOWN_MISSING = {
    "Rockström et al. 2009": "Nature article, not open access",
}
# Files in the folder that chapter 3 does not use
NOT_USED = {
    "European Environment Agency": "EEA is not used as reference in chapter 3",
    "Calisto Friant": "Not used in chapter 3",
}

YEAR = r"(?:19|20)\d{2}[a-z]?"
# A citation at the start of a file name: "Bletsis et al. 2024" or "Europees Parlement en Raad (2024)"
DESIGNATION_RE = re.compile(rf"^(.+?)\s*\(?({YEAR})\)?$")


def norm(text):
    """Lower case, no accents, '&', 'en' and 'and' unified, hyphens and extra spaces removed."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("-", " ").replace(".", " ")
    text = re.sub(r"\b(en|and)\b", "&", text)
    return re.sub(r"\s+", " ", text).strip()


def split_citation(citation):
    """'Bletsis et al. 2024' -> ('Bletsis et al.', '2024'). CLO codes have no year."""
    m = re.match(rf"^(.*) ({YEAR})$", citation)
    return (m.group(1), m.group(2)) if m else (citation, "")


def designation_key(name):
    """Normalise a citation for comparing with a file name. Initials and commas are dropped,
    so 'Bastein, T. & E. Rietveld 2024' and 'Bastein & Rietveld 2024' are equal."""
    name = re.sub(r"\b[A-Z]\.\s*", "", name).replace(",", " ")
    name = re.sub(r"\((" + YEAR + r")\)", r"\1", name)
    return norm(name)


# Citations
def load_citations():
    with open(CITED_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    citations = []
    for r in rows:
        if r["citation"] not in citations:
            citations.append(r["citation"])
    return citations


# File names
def file_designations(path):
    """The citations a file name starts with.

    'UNEP 2024a, IRP 2024, UNEP (2024a) Global ...' -> ['UNEP 2024a', 'IRP 2024']
    'CBS 2024a of  CBS 2024, Indicatoren ...'      -> ['CBS 2024a', 'CBS 2024']
    'Bastein, T. & E. Rietveld 2024, TNO (2015) ...' -> ['Bastein, T. & E. Rietveld 2024']
    """
    stem = re.sub(r"\s+", " ", path.stem)
    clo = re.match(r"^CLO \d{4}", stem)
    if clo:  # 'CLO 0621, Circular Material Use Rate ...'
        return [clo.group(0)]
    # The first citation runs up to the first year
    m = re.match(rf"^(.{{1,60}}?)\s*\(?({YEAR})\)?(?=[,\s]|$)", stem)
    if not m:
        return [stem.split(",")[0].strip()]
    names = [f"{m.group(1)} {m.group(2)}"]
    # More citations may follow, like ', IRP 2024' or ' of CBS 2024'
    rest = re.sub(r"^[,\s]*(?:of\s+)?", "", stem[m.end():])
    for part in re.split(r",\s*| of ", rest):
        dm = DESIGNATION_RE.match(part.strip())
        if not part.strip() or not dm or len(dm.group(1).split()) > 4 or "(" in part:
            break
        names.append(f"{dm.group(1)} {dm.group(2)}")
    return names


def publisher_year(path):
    """Year in brackets in the file name, like 'TNO (2015)'. Empty if there is none."""
    m = re.search(rf"\(({YEAR})\)", path.stem)
    return m.group(1) if m else ""


# Citation to file
def find_file(citation, files):
    """The file whose name starts with the citation, or with its corrected name. None if there is none.

    A file name can start with more than one citation, like 'UNEP 2024a, IRP 2024, ...'.
    """
    name = CORRECTIONS[citation][0] if citation in CORRECTIONS else citation
    key = designation_key(name)
    for path, names in files.items():
        if key in [designation_key(n) for n in names]:
            return path
    return None


# Groups
def classify(citation, path):
    if path is None:
        return "no_file", KNOWN_MISSING.get(citation, "no matching file")
    if citation in CONFLICTS:
        return "C", CONFLICTS[citation]
    if citation in CORRECTIONS:
        _, group, reason = CORRECTIONS[citation]
        return group, reason

    # The file name starts with the chapter citation, so the file was linked to it on purpose.
    # That makes it A, also when the year in brackets differs. The difference goes in the note.
    notes = []
    _, year = split_citation(citation)
    pub_year = publisher_year(path)
    if year and pub_year and pub_year[:4] != year[:4]:
        notes.append(f"chapter has {year}, year in brackets in the file name is {pub_year}")
    elif year and pub_year and pub_year != year and year[4:]:
        notes.append(f"letter after the year differs, year in brackets in the file name is {pub_year}")
    return "A", "; ".join(notes)


def main():
    citations = load_citations()
    files = {p: file_designations(p) for p in sorted(config.REFERENCES_DIR.rglob("*")) if p.is_file()}

    rows, used_files = [], set()
    for citation in citations:
        path = find_file(citation, files)
        group, note = classify(citation, path)
        rows.append({
            "citation": citation,
            "group": group,
            "file": path.name if path else "",
            "note": note,
        })
        if path:
            used_files.add(path)

    # Save the results
    config.STEP1_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["citation", "group", "file", "note"])
        writer.writeheader()
        writer.writerows(rows)

    unused = []
    for p in files:
        if p not in used_files:
            reason = next((v for k, v in NOT_USED.items() if p.name.startswith(k)), "")
            unused.append({"file": p.name, "note": reason})
    with open(UNUSED_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "note"])
        writer.writeheader()
        writer.writerows(unused)

    # Files used by more than one citation, like 'IenW et al. 2023' and 'IenW 2023'
    shared = {}
    for r in rows:
        if r["file"]:
            shared.setdefault(r["file"], []).append(r["citation"])
    shared = {f: c for f, c in shared.items() if len(c) > 1}

    counts = {g: sum(r["group"] == g for r in rows) for g in ("A", "B", "C", "no_file")}
    with_file = sum(1 for r in rows if r["file"])
    lines = [
        f"Files in references/: {len(files)}",
        f"  PDF: {sum(p.suffix.lower() == '.pdf' for p in files)}, "
        f"Excel: {sum(p.suffix.lower() in ('.xlsx', '.xls') for p in files)}",
        f"Files used by a citation: {len(used_files)}",
        f"Files not used by any citation: {len(unused)}",
        "",
        f"Citations in the chapter: {len(rows)}",
        f"Citations with a file: {with_file}",
        f"Citations without a file: {len(rows) - with_file}",
        f"Groups: A {counts['A']}, B {counts['B']}, C {counts['C']}, no_file {counts['no_file']}",
        "",
        "Files used by more than one citation:",
        *[f"  {', '.join(c)} -> {f}" for f, c in shared.items()],
        "",
        "Citations not in group A:",
        *[f"  [{r['group']}] {r['citation']} | {r['file'] or '-'} | {r['note']}" for r in rows if r["group"] != "A"],
        "",
        "Files not used by any citation:",
        *[f"  {u['file']}" + (f" | {u['note']}" if u["note"] else "") for u in unused],
    ]
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nsaved {OUT_FILE}\nsaved {UNUSED_FILE}\nsaved {SUMMARY_FILE}")


if __name__ == "__main__":
    main()

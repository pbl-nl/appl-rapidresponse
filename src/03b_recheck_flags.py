"""Step 3b: check the evidence and the numbers of the extracted facts again, with less strict rules.

No API calls. Reads outputs/03_extraction/facts.jsonl and the step 2 passages, and leaves both
as they are. Writes facts_checked.jsonl to the same folder, with the new flags in place of the
old evidence and number flags. This is the facts file for step 4. The counts before and after
are only printed.

Usage:
    python src/03b_recheck_flags.py               # recheck all facts
    python src/03b_recheck_flags.py --samples 5   # also print 5 samples per flag reason
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Step 3 file name starts with a digit, so it is loaded by path
spec = importlib.util.spec_from_file_location("extraction", Path(__file__).with_name("03_extraction.py"))
extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extraction)

FACTS_FILE = config.STEP3_DIR / "facts.jsonl"
PASSAGES_FILE = config.STEP2_DIR / "passages.jsonl"
CHECKED_NAME = "facts_checked.jsonl"
# Old step 3 flags, replaced by the new ones
REPLACED = {"evidence_found", "number_found", "multiple_quantities", "quantity_needs_review"}
# Share of evidence words that must be in the passage in the same order
SAME_TEXT = 0.9

FRACTION_WORD = (r"half|helft|derde|kwart|vierde|vijfde|tiende|honderdste|"
                 r"third|quarter|fifth|tenth|dozen|tientallen|honderden|duizenden")
NUMBER_WORDS = re.compile(rf"\b(?:{extraction.NUMBER_WORD}|{FRACTION_WORD})\b", re.I)
# A number that is not part of a word, so the 2 in 'CO2' is skipped
NUMBER = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")

REASONS = ["evidence_not_found", "number_not_found", "number_not_in_evidence", "survey_count",
           "other_number_check", "scenario", "language"]


def numbers_in(text):
    return NUMBER.findall(extraction.normalise(text))


def has_number(number, text):
    """True if the number stands on its own in the text. A dot and a comma count as the same.

    A unit right after the number is fine, like '800bn' or '2010s', but '1' is not found in '51' or '1,5'.
    """
    parts = re.split(r"[.,]", number)
    pattern = r"(?<![\w.,])" + r"[.,]".join(re.escape(p) for p in parts) + r"(?!\d|[.,]\d)"
    return bool(re.search(pattern, extraction.normalise(text)))


def has_word(word, text):
    return bool(re.search(rf"\b{re.escape(word)}\b", extraction.normalise(text)))


def check_numbers(record):
    """Return (number_ok, numbers in the fact that are not in the evidence).

    Every number and number word in the number field must be in the evidence and in the fact.
    A range like '10 en 60' or a sign like '€10' is fine. If the number field holds no number,
    like 'N.v.t.', only the numbers of the fact are checked.
    """
    number, fact, evidence = record["number"], record["fact"], record["evidence"]
    digits = numbers_in(number)
    words = [w.lower() for w in NUMBER_WORDS.findall(number)]
    number_ok = (all(has_number(d, evidence) and has_number(d, fact) for d in digits)
                 and all(has_word(w, evidence) and has_word(w, fact) for w in words))

    # The year field may come from a table header, so that year does not need to be in the evidence
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", record.get("year") or ""))
    missing = [d for d in numbers_in(fact) if d not in years and not has_number(d, evidence)]
    return number_ok, missing


def padded_words(text):
    return f" {extraction.words_only(text)} "


def check_evidence(record, passage_text):
    """True if the evidence comes from the passage.

    Passes when the words are the same and only punctuation or spaces differ, when the evidence
    is made of pieces of the passage (like a table header and one cell), or when almost all words
    are the same (a broken word or a footnote number fixed). Evidence shortened with dots fails,
    and every number in the evidence must be in the passage.
    """
    if record["evidence_found"]:
        return True
    evidence = record["evidence"]
    if re.search(r"\.\.\.|…", evidence):
        return False
    if not all(has_number(d, passage_text) for d in numbers_in(evidence)):
        return False

    passage_words = padded_words(passage_text)
    if padded_words(evidence) in passage_words:
        return True
    pieces = [p for p in re.split(r"(?<=[.!?])\s+|\n+|[:;=]|\s/\s", evidence) if extraction.words_only(p)]
    if pieces and all(padded_words(p) in passage_words for p in pieces):
        return True

    words = extraction.words_only(evidence).split()
    if not words:
        return False
    matcher = SequenceMatcher(None, passage_words.split(), words, autojunk=False)
    same = sum(block.size for block in matcher.get_matching_blocks())
    return same / len(words) >= SAME_TEXT


def survey_mismatch(record):
    counts = record.get("survey_counts")
    if not counts:
        return False
    rate = float(re.match(r"\d+(?:[.,]\d+)?", record["number"])[0].replace(",", "."))
    return (not counts["total"] or counts["respondents"] > counts["total"]
            or abs(100 * counts["respondents"] / counts["total"] - rate) > 0.5)


def recheck(facts, passages):
    # Same order as in the file, grouped by passage for the checks that compare facts
    by_passage = {}
    for r in facts:
        by_passage.setdefault(r["passage_id"], []).append(r)

    checked = []
    for records in by_passage.values():
        results = [check_numbers(r) for r in records]
        new = [{**r, "number_found": ok, "multiple_quantities": bool(missing)}
               for r, (ok, missing) in zip(records, results)]
        # The step 3 checks for years and survey counts, run with the new number results
        review = extraction.quantity_review_indices(new)

        for i, r in enumerate(records):
            number_ok, missing = results[i]
            evidence_ok = check_evidence(r, passages[r["passage_id"]])
            reasons = []
            if not evidence_ok:
                reasons.append("evidence_not_found")
            if not number_ok:
                reasons.append("number_not_found")
            if missing:
                reasons.append("number_not_in_evidence")
            if survey_mismatch(r):
                reasons.append("survey_count")
            if i in review and number_ok and not missing and not survey_mismatch(r):
                reasons.append("other_number_check")
            if r["scenario_needs_review"]:
                reasons.append("scenario")
            if r["language_mismatch"]:
                reasons.append("language")
            record = {k: v for k, v in r.items() if k not in REPLACED}
            checked.append({**record, "evidence_ok": evidence_ok, "number_ok": number_ok,
                            "numbers_not_in_evidence": missing,
                            "flag_reasons": reasons, "flagged": bool(reasons)})
    return checked


def print_summary(facts, checked):
    total = len(facts)
    old = sum(extraction.is_flagged(r) for r in facts)
    new = sum(r["flagged"] for r in checked)
    reasons = Counter(reason for r in checked for reason in r["flag_reasons"])
    lines = [
        "Recheck of the step 3 flags",
        "",
        f"facts: {total}",
        f"flagged before: {old} ({old / total:.1%})",
        f"flagged now: {new} ({new / total:.1%})",
        "",
        "Before, per check",
        f"  evidence not found: {sum(not r['evidence_found'] for r in facts)}",
        f"  number not found: {sum(r['number_found'] is False for r in facts)}",
        f"  more than one number: {sum(bool(r['multiple_quantities']) for r in facts)}",
        "",
        "Now, per reason (a fact can have more than one)",
    ]
    lines += [f"  {reason}: {reasons[reason]}" for reason in REASONS]
    lines += ["", f"chart pages, not counted as flagged: {sum(r['needs_chart_review'] for r in checked)} facts"]
    print("\n".join(lines))


def print_samples(checked, n):
    random.seed(1)
    for reason in REASONS:
        group = [r for r in checked if reason in r["flag_reasons"]]
        if not group:
            continue
        print(f"\n--- {reason} ({len(group)}) ---")
        for r in random.sample(group, min(n, len(group))):
            print(f"{r['fact_id']}  {r['source']} p{r['page']}")
            print(f"  fact:     {r['fact']}")
            print(f"  number:   {r['number']}")
            if r["numbers_not_in_evidence"]:
                print(f"  missing:  {', '.join(r['numbers_not_in_evidence'])}")
            print(f"  evidence: {' '.join(r['evidence'].split())[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=0, help="print this many samples per flag reason")
    args = parser.parse_args()

    with open(FACTS_FILE, encoding="utf-8") as f:
        facts = [json.loads(line) for line in f]
    passages = {}
    with open(PASSAGES_FILE, encoding="utf-8") as f:
        for line in f:
            passage = json.loads(line)
            passages[passage["passage_id"]] = passage["text"]
    missing = {r["passage_id"] for r in facts} - set(passages)
    if missing:
        sys.exit(f"{len(missing)} passages of the facts are not in passages.jsonl. "
                 f"Step 2 and step 3 do not belong together.")
    checked = recheck(facts, passages)

    with open(config.STEP3_DIR / CHECKED_NAME, "w", encoding="utf-8") as f:
        for r in checked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print_summary(facts, checked)
    print(f"\nfacts -> {config.STEP3_DIR / CHECKED_NAME}")

    if args.samples:
        print_samples(checked, args.samples)


if __name__ == "__main__":
    main()

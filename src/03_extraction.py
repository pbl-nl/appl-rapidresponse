"""Step 3: the LLM reads each passage and writes facts with evidence.

Every fact is checked. The evidence must be in the passage, and the number must be in
the evidence and in the fact. Facts with a problem get one retry. Facts that still fail
are kept but flagged. Pages with charts are listed in chart_review.md for a manual check.
Results go to outputs/03_extraction/. OPENAI_API_KEY and LLM_MODEL come from .env.

Usage:
    python src/03_extraction.py --dry-run                      # no API calls, count passages and tokens
    python src/03_extraction.py --dry-run --passage P00012     # show the prompt for one passage
    python src/03_extraction.py --passage P00012               # one passage
    python src/03_extraction.py --file Verpact --page 7        # all passages of one page
    python src/03_extraction.py --workers 4 --yes              # up to four passages in parallel
    python src/03_extraction.py                                # all passages (asks first)
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

PASSAGES_FILE = config.STEP2_DIR / "passages.jsonl"
FACTS_FILE = config.STEP3_DIR / "facts.jsonl"
DONE_FILE = config.STEP3_DIR / "done_passages.txt"

# The form of the answer. Strict mode needs every field, so 'none' is an empty string.
FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "number": {"type": "string"},
                    "year": {"type": "string"},
                    "quantity_status": {"type": "string", "enum": ["observed", "projected", "unspecified"]},
                    "evidence": {"type": "string"},
                },
                "required": ["fact", "number", "year", "quantity_status", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You extract facts from a passage of a source document about the environment, \
resources and the circular economy. Extract all substantive claims supported by the passage.

For each fact give:
- fact: one fact in your own words, one piece of information per fact, written in {language}. \
The fact must be understandable on its own, so name the subject, place and unit \
(Dutch example: "De invoer van grondstoffen in Nederland groeide met 11 procent.").
- number: the number exactly as written in the passage (for example "11", "2,5", "4,5 keer"). \
Copy the number from the evidence, including spelled-out forms such as "veertien". \
Use that same wording in the fact. Footnote and reference markers are never quantities. \
Split independent measurements into separate claims even if one sentence supplies all of them.
Keep a genuine interval as one range. Years, labels such as CO₂ and necessary scenario conditions
can accompany a measurement. Put survey counts in the evidence; state the percentage alone in the fact.
Empty if the fact has no number. One main quantity per fact.
- For a change over time, return a separate fact for each endpoint. Associate each value \
with its own year and subject. Keep the unit, approximations, scenario assumptions and \
forecast wording in the fact. For example, if a 2019 share of 50% is expected to reach 57% \
in 2030, emit 50% for 2019 (observed) and 57% for 2030 (projected). Include a separately \
stated growth rate as its own fact. Do not calculate missing quantities.
- Never concatenate values. A truncated count such as "218 out of" is incomplete: omit that \
count, but keep any independently stated percentage. Use one source-form quantity in both \
number and fact, including attached units such as "2.3Mt". Keep facts in the source language.
- year: the year of this quantity. Use a period only for a quantity applying to that whole \
period. Leave empty if unknown; the source publication year is not a measurement year.
Do not duplicate a value across years because the ranking was unchanged or a share was \
approximately the same. Report only the year for which that value was supplied. For projections \
write "naar verwachting" or "is projected" in the claim itself. Make the evidence long enough to support it.
- quantity_status: observed for reported measurements or historical baselines, projected for \
forecasts or scenario outputs, unspecified for other or unclear cases.
For projections, copy every applicable explicit scenario condition verbatim into the fact \
and include its source sentence in evidence. If the next sentence states the resulting change, \
carry the same condition into that change. Do not attach it to a historical baseline.
If an assumption is immediately followed by "Dit levert", "This yields" or a similar result, \
include that assumption in each resulting total and annual estimate. Identify whose potential \
or population it concerns. Keep the result's number field separate from the assumed percentage.
- evidence: the exact text from the passage that supports the fact, copied word for word. \
The evidence is one continuous piece of the passage, with the original punctuation. \
Never insert an ellipsis or join fragments. Copy a full sentence or table row.

Language: write each fact in the same language as its evidence (the language of the passage). \
Use Dutch for Dutch evidence and English for English evidence. Never translate.

Tables: only report a value when one continuous piece of evidence includes the row label, column \
heading, value and any applicable unit, and their relationship is unambiguous. Skip cells \
whose meaning depends on missing headings or lost layout. Never guess a column by position.
A color legend alone gives no row or cell rating. Omit ratings unless the cell value is in text.
For numeric tables copy the full header and all intervening rows through the target row verbatim.
For policy tables distinguish implementation abroad from the proposed jurisdiction.
Use deposit or statiegeld for the object actually named; do not introduce bottles.
Before returning, check each sentence for additional supported claims, including comparisons, \
majorities and geographic qualifiers. Extract each distinct finding.
Study design is in scope: extract explicit counts of interviews and roundtables separately, and \
the full dates a consultation opened and closed, even under a Methods or Introduction heading.
For a survey percentage, put only the percentage in number. Copy the reported numerator and \
denominator into the evidence, with the group and response. Keep different questions and subgroups \
distinct. Do not reconstruct a missing question from a fragment or infer missing counts.
For an event spanning two full dates, emit one fact with its opening date and one with its \
closing date. Copy the complete original sentence for each. A date itself is one value.
Skip contents entries, page numbers, running headers, bibliography entries, contact details \
and publisher boilerplate. On mixed pages, extract only substantive claims.
If the passage has no facts, return an empty list."""

# Second prompt for facts that failed the check. The facts to fix are added at the end.
CORRECTION = """Correct only the following records using the original passage.
Use the supplied original_index for every replacement, including each split claim.
Return only replacements for those indices in the source language.
For each endpoint return only its own value and year, explicitly stating whether it is a forecast.
Example number fields: '50%' and '57%' in separate records. Do not repeat the range in either fact.
Never infer a forecast status for the historical baseline. Give the initial observation its own
standalone sentence, with no later year, forecast or assumption.
For a change spanning two years, use the full period as its year.
Never infer a historical percentage from a comparison of rankings.
Drop unsupported duplicates of endpoints, but retain each supported projection and copy its
applicable scenario_conditions verbatim into the standalone claim.
Split independent measurements in the claim, including spelled-out values. Keep a genuine range together.
Copy the evidence as one continuous piece of the original text, with its punctuation. Never insert ellipses.
"""

# Prompt for facts written in another language than their evidence
LANGUAGE_FIX = """Rewrite only the fact fields of these records in the language of their source evidence.
Preserve order, meaning and all other fields.
"""


# Passages
def load_passages():
    with open(PASSAGES_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def needs_chart_review(passage):
    """True for PDF passages where chart values lost their labels. These need a manual check."""
    if Path(passage["source_file"]).suffix.lower() != ".pdf":
        return False
    lines = [line.strip() for line in passage["text"].splitlines() if line.strip()]
    value = r"[<>≤≥+−-]?\d+(?:[.,]\d+)?\s*%"
    detached = sum(bool(re.fullmatch(rf"{value}(?:\s+{value})*", line)) for line in lines)
    if detached >= 2:
        return True
    # Years in one block and numbers in the next block, so it is not clear which number goes with which year
    if re.search(r"(?m)^(?:(?:19|20)\d{2}[ \t]*\n\s*){3,}(?=[−-]?\d+(?:[.,]\d+)?[ \t]*$)",
                 passage["text"]):
        return True
    caption = re.search(r"\b(?:figuur|figure|grafiek|chart)\s+\d+", passage["text"], re.I)
    return bool(caption and sum(bool(re.fullmatch(r"[−-]?\d+(?:[.,]\d+)?", line))
                                for line in lines) >= 3)


def write_chart_review(passages, output_dir):
    """Write chart_review.md with a link to every page that needs a manual check."""
    pages = {}
    for passage in passages:
        if needs_chart_review(passage):
            key = (passage["source_file"], passage["page"])
            pages.setdefault(key, []).append(passage["passage_id"])
    path = output_dir / "chart_review.md"
    if not pages and not path.exists():
        return
    sources = {p.name: p for p in config.REFERENCES_DIR.rglob("*") if p.is_file()}
    lines = ["# Chart and table pages for visual review", "",
             "Verify the label and value together on the original page, including passages with zero facts.", ""]
    for (filename, page), ids in pages.items():
        label = f"{filename} (page {page})"
        if filename in sources:
            target = quote(os.path.relpath(sources[filename], output_dir))
            label = f"[{label}]({target}#page={page})"
        lines.append(f"- {label} - {', '.join(ids)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extraction_text(text):
    """Remove tables that only have labels, because their values were shown as colours in the PDF."""
    blocks = re.split(r"\n[ \t]*\n", text)
    start = None
    keep = []
    for block in blocks:
        if re.match(r"^(?:Tabel|Table)\s+\d+\b", block, re.I):
            start = len(keep)
        if start is not None and re.match(r"^(?:Wit|White)\s*:", block, re.I):
            labels = keep[start + 1:]
            if labels and all(re.fullmatch(r"[^\d.!?:;\n]+", label)
                              and len(label.split()) <= 6 for label in labels):
                del keep[start:]
                start = None
                continue
        keep.append(block)
    return "\n\n".join(keep).strip()


def prepare_passage(passage):
    review = needs_chart_review(passage)
    text = extraction_text(passage["text"])
    if review:
        # Loose labels and axis numbers can not support a fact. Keep only normal sentences
        # and captions with a total, a unit and a year.
        lines = []
        for line in text.splitlines():
            words = re.findall(r"[^\W\d_]+", line)
            narrative = len(words) >= 12 and re.search(r"(?<!\.)[.!?](?![.\d])|\b(?:was|were|showed|zijn|volgens)\b", line, re.I)
            caption = re.match(r"^\s*\d+(?:[.,]\d+)?\s*(?:Mt|kt|tonnes?)\b", line, re.I)
            if narrative or (caption and len(words) >= 5 and re.search(r"\b(?:19|20)\d\d\b", line)):
                lines.append(line)
        text = "\n\n".join(lines)
    return {**passage, "source_text": passage["text"], "text": text, "needs_chart_review": review}


def table_facts(text):
    """Facts from two known Dutch policy tables, made without the LLM. None for other passages."""
    blocks = text.strip().split("\n\n")
    if len(blocks) < 3:
        return None
    title, header = blocks[:2]
    counts = re.fullmatch(r"Tabel \d+ - Beoordeling instrumenten (voorgenomen|vastgesteld) beleid", title)
    if counts and header == "Normerend Beprijzend Stimulerend":
        facts = []
        for i, block in enumerate(blocks[2:], 2):
            if i == len(blocks) - 1 and block.startswith("Wit:") and "Groen:" in block:
                break
            row = block.splitlines()
            if len(row) != 4 or not all(re.fullmatch(r"\d+|[−–-]", v) for v in row[1:]):
                return None
            for kind, value in zip(("normerend", "beprijzend", "stimulerend"), row[1:]):
                if not value.isdigit():
                    continue
                fact = (f"Voor '{row[0]}' bedraagt het aantal instrumenten van het type "
                        f"'{kind}' in het {counts[1]} beleid {value}.")
                facts.append(dict(fact=fact, number=value, year="", quantity_status="unspecified",
                                  evidence="\n\n".join(blocks[:i + 1])))
        return facts

    if (not title.startswith("Tabel ") or blocks[1:4] != ["Omschrijving instrument Type beleidsinstrument",
            "Al geïmplementeerd in buitenland?", "Schaalniveau"]):
        return None
    # Only complete rows are used. The last row can go on to the next page.
    body = "\n\n".join(blocks[4:])
    end = re.compile(r"\s+Norm\s+(Ja,\s*([^\n]+?)|Nee)\s+(EU/NL|NL|EU)(?=\n\n|$)")
    facts = []
    start = 0
    for match in end.finditer(body):
        description = body[start:match.start()].strip()
        if not description or re.search(r"\b(?:Norm|Subsidie|Belasting)\s+(?:Ja|Nee)\b", description):
            return None
        evidence = body[start:match.end()].strip()
        description = re.sub(r"\s+\d+$", "", description)
        instrument = " ".join(description.split())
        status = (f"volgens het rapport ingevoerd in {match[2]}" if match[2]
                  else "volgens het rapport niet in het buitenland ingevoerd")
        duration = re.search(r"\b(?:\d+|een|twee|drie|vier|vijf|zes|zeven|acht|negen|tien) jaar\b",
                             instrument, re.I)
        facts.append(dict(fact=f"De maatregel '{instrument}' is {status}.",
                          number=duration[0] if duration else "", year="",
                          quantity_status="unspecified", evidence=evidence))
        facts.append(dict(fact=f"Het rapport noemt {match[3]} als schaalniveau voor '{instrument}'.",
                          number="", year="", quantity_status="unspecified", evidence=evidence))
        start = match.end()
    tail = body[start:].strip()
    if tail and ("\n" in tail or not tail.startswith("Verbod op ") or tail.endswith(".")):
        return None
    return facts or None


# LLM call
def build_messages(passage):
    language = config.FACT_LANGUAGE or "the language of the passage"
    passage_language = likely_language(passage['text'])
    if language == "the language of the passage" and passage_language:
        language = {"nl": "Dutch (Nederlands)", "en": "English"}[passage_language]
    instructions = INSTRUCTIONS.format(language=language)
    if passage.get("needs_chart_review"):
        instructions += ("\nChart labels and values may be detached. Only extract quantities directly "
                         "associated with their labels in prose or a self-contained caption. "
                         "Leave ambiguous chart values for visual review.")
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"Source: {passage['citation']}, page {passage['page']}\n\n"
                                    f"Passage:\n{passage['text']}"},
    ]


EVENT_DATE = r"(?:[1-9]|[12]\d|3[01])\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
EVENT_PERIOD = re.compile(
    rf"(?P<event>[^.!?\n]+?) ran (?:between|from) (?P<start>{EVENT_DATE}) "
    rf"(?:and|to) (?P<end>{EVENT_DATE})(?:\s+(?P<place>on [^.!?\n]+))?[.]?", re.I
)


def split_event_dates(facts, text):
    """Split a period like 'ran from 1 May 2022 to 2 August 2022' into a start fact and an end fact."""
    result = []
    for fact in facts:
        evidence = fact["evidence"].strip()
        period = EVENT_PERIOD.fullmatch(evidence)
        if not (period and evidence in text and period["start"] in fact["number"]
                and period["end"] in fact["number"]):
            result.append(fact)
            continue
        location = " " + period["place"] if period["place"] else ""
        for date, verb in [(period["start"], "opened"), (period["end"], "closed")]:
            result.append({**fact, "fact": f"{period['event']} {verb} on {date}{location}.",
                           "number": date, "year": date[-4:]})
    return result


class InvalidExtractionResponse(ValueError):
    """The API returned an incomplete or invalid extraction, which can be retried."""


def response_facts(response, schema):
    """Validate the complete response before any facts can be saved."""
    if not response.choices:
        raise InvalidExtractionResponse("response has no choices")
    choice = response.choices[0]
    content = choice.message.content
    context = f"finish_reason={choice.finish_reason}, chars={len(content or '')}"
    if choice.message.refusal:
        raise InvalidExtractionResponse(f"response refused ({context})")
    if choice.finish_reason != "stop":
        raise InvalidExtractionResponse(f"response did not finish normally ({context})")
    if not content:
        raise InvalidExtractionResponse(f"empty response ({context})")
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidExtractionResponse(f"invalid JSON: {error} ({context})") from error
    if not isinstance(answer, dict) or set(answer) != {"facts"} or not isinstance(answer["facts"], list):
        raise InvalidExtractionResponse("response must contain a facts array")
    fields = schema["properties"]["facts"]["items"]["properties"]
    for i, fact in enumerate(answer["facts"]):
        if not isinstance(fact, dict) or set(fact) != set(fields):
            raise InvalidExtractionResponse(f"fact {i} has missing or unexpected fields")
        for name, field in fields.items():
            expected = int if field["type"] == "integer" else str
            if type(fact[name]) is not expected or ("enum" in field and fact[name] not in field["enum"]):
                raise InvalidExtractionResponse(f"fact {i} has invalid {name}")
    return answer["facts"]


def extract_facts(client, passage, correction=None, indexed=False):
    """Ask for facts, retrying API errors and invalid or incomplete responses."""
    import openai
    messages = build_messages(passage)
    if correction:
        messages.append({"role": "user", "content": correction})
    schema = FACT_SCHEMA
    if indexed:
        schema = deepcopy(schema)
        item = schema["properties"]["facts"]["items"]
        item["properties"]["original_index"] = {"type": "integer"}
        item["required"].append("original_index")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)
    last_error = None
    for attempt in range(1, config.EXTRACTION_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "facts", "strict": True, "schema": schema}},
            )
            if response.usage:
                usage.prompt_tokens += response.usage.prompt_tokens
                usage.completion_tokens += response.usage.completion_tokens
            facts = response_facts(response, schema)
            return split_event_dates(facts, passage["text"]), usage
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError, InvalidExtractionResponse) as error:
            last_error = error
            print(f"    {passage['passage_id']}: {type(error).__name__}, "
                  f"try {attempt} of {config.EXTRACTION_RETRIES}: {error}")
            if attempt == config.EXTRACTION_RETRIES:
                break
            wait = 2 ** attempt
            print(f"    waiting {wait} s before retry")
            time.sleep(wait)
    raise RuntimeError(f"no valid answer for {passage['passage_id']} after "
                       f"{config.EXTRACTION_RETRIES} tries; passage remains unfinished") from last_error


# Automatic check
def normalise(text):
    """Text for comparing: same quotes, same hyphens, soft hyphens and line breaks removed,
    one space between words, lower case."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = re.sub(r"[‘’‚‛`´]", "'", text)
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def words_only(text):
    """Only the words and numbers of a text, one space between them.

    Punctuation, quote marks, bullets and odd characters are dropped, so 'afvalfase,' and
    'afvalfase.' or "'narrow the loop'" and "narrow the loop" become the same.
    """
    return " ".join(re.findall(r"[\w%€]+", normalise(text)))


def restore_table_evidence(evidence, passage_text):
    """Put back the table rows the LLM left out between the header and the row it used."""
    parts = re.split(r"\n\s*\n", evidence.strip())
    if len(parts) != 2:
        return evidence
    header, row = parts
    lines = row.splitlines()
    if len(lines) < 3 or len(header.split()) != len(lines) - 1:
        return evidence
    numeric = r"[-+]?\d+(?:[.,]\d+)?%?"
    if not all(re.fullmatch(numeric, cell.strip()) for cell in lines[1:]):
        return evidence
    blocks = list(re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]+)*", passage_text))
    matches = []
    for i, block in enumerate(blocks):
        if block.group().strip() != header:
            continue
        for following in blocks[i + 1:]:
            candidate = following.group().strip()
            cells = candidate.splitlines()
            if (len(cells) != len(lines) or not any(c.isalpha() for c in cells[0])
                    or not all(re.fullmatch(numeric, cell.strip()) for cell in cells[1:])):
                break
            if candidate == row:
                matches.append(passage_text[block.start():following.end()].strip())
    return matches[0] if len(matches) == 1 else evidence


# Number patterns. Decimal and thousand separators stay part of the number.
DIGITS = r"[+-]?\d+(?:[.,]\d+)*(?:[⁄/]\d+)?"
BOUND = rf"(?:[€$£]\s*)?{DIGITS}(?:\s*%)?"
RANGE = re.compile(rf"(?<!\w){BOUND}\s*(?:[-–—]|\bto\b|\btot\b)\s*{BOUND}(?!\d)", re.I)
NUMBER_WORD = (r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
               r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
               r"forty|fifty|sixty|seventy|eighty|ninety|nul|één|twee|drie|vier|vijf|zes|"
               r"zeven|acht|negen|tien|elf|twaalf|dertien|veertien|vijftien|zestien|"
               r"zeventien|achttien|negentien|twintig|dertig|veertig|vijftig|zestig|zeventig|"
               r"tachtig|negentig")
INTERVAL_BOUND = rf"(?:{BOUND}|(?:{NUMBER_WORD})\b)"
BETWEEN = re.compile(rf"\b(?:between|tussen)\s+{INTERVAL_BOUND}\s+(?:and|en)\s+{INTERVAL_BOUND}", re.I)
MEASURE = re.compile(rf"(?<!\w)(?:{DIGITS}|(?:{NUMBER_WORD})(?:[- ](?:{NUMBER_WORD}))?\b)", re.I)
MONTH = (r"jan(?:uary|uari)?|feb(?:ruary|ruari)?|maart|march|apr(?:il)?|may|mei|jun[ei]?|"
         r"july|juli|augustus|august|september|o[ck]tober|november|december")
DATE = re.compile(rf"\b\d{{1,2}}\s+(?:{MONTH})\s+\d{{4}}\b", re.I)


def is_range(number):
    """True if the number is one range, like '10-20%' or 'tussen 5 en 10'."""
    return bool(re.fullmatch(rf"(?:circa |about |approximately |ongeveer )?(?:{RANGE.pattern}|{BETWEEN.pattern})"
                             r"(?:\s+[^\d,;]+)?", number, re.I))


def quantity_count(text, year="", main_value=""):
    """Count the numbers in a text. Dates, ranges, years and labels like 'tabel 3' are not counted."""
    text = unicodedata.normalize("NFKC", text).casefold()
    # A date can be the number itself, or only say when something happened
    date_marker = " " if main_value and not DATE.search(main_value) else " 0 "
    text = DATE.sub(date_marker, text)
    text = re.sub(r"\bten (?=behoeve|opzichte|minste|aanzien|gevolge|gunste|tijde)\b", "", text)
    text = re.sub(r"\b(?:eu\d+\+\d+|\d+-onder-\d+-kap|(?:sectoren|sectors)\s+\d+\s*"
                  r"(?:[–—-]|tot en met|to)\s*\d+)\b", " ", text)
    if main_value:
        text = re.sub(rf"\b(?:de afgelopen|the past|the last) (?:\d+|{NUMBER_WORD}) "
                      r"(?:years|months|jaar|jaren|maanden)\b", " ", text)
    # Years and labels like 'figuur 2' are not numbers to count
    for y in re.findall(r"\b(?:19|20)\d{2}\b", year):
        text = re.sub(rf"\b{y}\b(?!\s*%)", " ", text)
    text = re.sub(r"\b(?:in|since|sinds|by|vóór|voor|between|tussen|from|van|to|tot|and|en|aan)\s+"
                  r"(?:19|20)\d{2}\b(?!\s*%)", " ", text)
    text = re.sub(r"\b(?:figuur|figure|tabel|table|article|artikel)\s+\d+(?:\.\d+)*", " ", text)
    # 'from 50 to 57' holds two separate numbers
    text = re.sub(rf"\b(from|van)\s+({BOUND})\s+(?:to|naar)\s+", r"\1 \2 ; ", text)
    text = BETWEEN.sub(" 0 ", text)
    text = RANGE.sub(" 0 ", text)
    text = re.sub(rf"\b(?:{NUMBER_WORD}|een)\s+(?:out of|op de|in)\s+(?:{NUMBER_WORD}|\d+)\b",
                  " 0 ", text)
    return len(MEASURE.findall(text))


def check_fact(fact, passage_text):
    """Return (evidence_found, number_found).

    The evidence must be in the passage, only spaces can differ. The number must be in the
    evidence and in the fact, and it must be one number (a range also counts as one).
    """
    exact = " ".join(fact["evidence"].split())
    evidence_found = bool(re.search(r"\w", exact)) and exact in " ".join(passage_text.split())
    evidence = normalise(fact["evidence"])
    number = re.sub(r"\s+%", "%", normalise(fact["number"]))
    # The number must stand on its own: '1' is not found inside '51' or '1,5'
    whole = re.compile(rf"(?<!\w)(?<!\d[.,]){re.escape(number)}(?!\w|[.,]\d)")
    bundled = re.search(r"\d.*(?:[,;]\s+|\b(?:naar|tot|to|and|en)\b)\D*\d", number)
    incomplete = re.search(r"\b(?:out of|van de|van het)\s*$", number)
    claim = re.sub(r"\s+%", "%", normalise(fact["fact"]))
    evidence = re.sub(r"\s+%", "%", evidence)
    claim_match = whole.search(claim)
    mass = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*mt", number)
    if mass and not claim_match:
        value = "[.,]".join(re.split(r"[.,]", mass[1]))
        claim_match = re.search(rf"(?<![\d.,]){value}\s*(?:mt|million tonnes|miljoen ton)\b", claim)
    number_found = not number or bool((not bundled or is_range(number)) and not incomplete and
                                      whole.search(evidence) and claim_match)
    return evidence_found, number_found


def likely_language(text):
    """Guess if a text is Dutch (nl) or English (en) from common short words. None if not clear."""
    words = set(re.findall(r"[^\W\d_]+", text.casefold()))
    dutch = len(words & {"de", "het", "een", "van", "voor", "met", "bij", "zijn", "wordt",
                         "door", "naar", "op", "aan", "uit", "tot", "en", "om", "te"})
    english = len(words & {"the", "a", "an", "of", "for", "with", "by", "from", "and",
                           "there", "implemented", "been", "system", "on", "which", "global", "production",
                           "are", "was", "were", "has", "have", "at", "to", "its"})
    if dutch >= english + 2:
        return "nl"
    if english >= dutch + 2:
        return "en"
    return None


# Records
def done_passages(path=DONE_FILE):
    if not path.exists():
        return set()
    return set(path.read_text(encoding="utf-8").split())


def next_fact_number(path=FACTS_FILE):
    if not path.exists():
        return 1
    with open(path, encoding="utf-8") as f:
        ids = [json.loads(line)["fact_id"] for line in f]
    return max((int(i[1:]) for i in ids), default=0) + 1


SCENARIO_MARKER = re.compile(
    r"\b(?:^als(?=\s.+?,\s*dan\b)|wanneer|indien|mits|onder de (?:aanname|voorwaarde) dat|uitgaande van|"
    r"verondersteld dat|assuming(?: that)?|assumed that|provided that|on condition that|if|under the assumption that)\b", re.I
)

DIRECT_RESULT = re.compile(r"^(?:dit levert|dit resulteert in|this yields|this results in)\b", re.I)


def split_sentences(text):
    """Sentences of a text. A blank line also ends a sentence."""
    return re.split(r"(?<=[.!?])\s+|\n\s*\n", text)


def overlaps(words_a, words_b):
    """True if one text is part of the other. Both come from words_only()."""
    return f" {words_a} " in f" {words_b} " or f" {words_b} " in f" {words_a} "


def attach_result_assumption(fact, text):
    """For a result like 'Dit levert ...', put the sentence with its assumption in front of the fact."""
    if fact.get("quantity_status") != "projected" or not fact["number"]:
        return fact
    evidence_words = words_only(fact["evidence"])
    if not evidence_words:
        return fact
    sentences = split_sentences(text)
    matches = []
    for assumption, result in zip(sentences, sentences[1:]):
        if not DIRECT_RESULT.match(result) or not SCENARIO_MARKER.search(assumption):
            continue
        if not overlaps(evidence_words, words_only(result)):
            continue
        if not check_fact({**fact, "evidence": result}, result)[1]:
            continue
        if words_only(assumption) in words_only(fact["fact"]):
            continue
        matches.append((assumption, result))
    if len(matches) == 1:
        assumption, result = matches[0]
        return {**fact, "fact": assumption + " " + fact["fact"], "evidence": assumption + " " + result}
    return fact


def projection_conditions(fact, passage_text):
    """The conditions (like 'als', 'if', 'assuming') in the evidence sentence of a projected fact."""
    if fact.get("quantity_status") != "projected":
        return []
    # Only look at the sentences that hold the evidence
    sentences = split_sentences(passage_text)
    evidence_words = words_only(fact["evidence"])
    if not evidence_words:
        return []
    conditions = []
    for i, sentence in enumerate(sentences):
        words = words_only(sentence)
        if not words or not overlaps(evidence_words, words):
            continue
        if i and re.match(r"^(dit levert|dit resulteert in|this (?:yields|results in)|"
                          r"dit is een (?:toename|afname)|this (?:is|represents) an? (?:increase|decrease))\b",
                          sentence, re.I):
            sentence = sentences[i - 1] + " " + sentence
        for match in SCENARIO_MARKER.finditer(sentence):
            clause = re.split(r"[,;.!?](?!\d)", sentence[match.start():], maxsplit=1)[0].strip()
            clause = re.sub(r"\s*\((?:figuur|figure|tabel|table) [\d.]+\)$", "", clause, flags=re.I)
            if clause and clause not in conditions:
                conditions.append(clause)
    return conditions


SURVEY_RESPONSE = re.compile(
    r"(?<![\d.,])(?P<percent>\d+(?:[.,]\d+)?\s*%)"
    r"(?:\s+of (?:all |the )?respondents)?\s*[(–—-]\s*"
    r"(?P<count>\d+)\s+(?:out of|van de)\s+(?P<total>\d+)\b", re.I
)


def survey_response(fact):
    """The survey counts next to the percentage, like 134 out of 331. None if not clear."""
    rate = re.match(r"^(\d+(?:[.,]\d+)?\s*%)", fact["number"].strip())
    if not rate:
        return None
    percent = re.sub(r"\s+", "", rate[1])
    matches = {(int(m["count"]), int(m["total"])) for m in SURVEY_RESPONSE.finditer(fact["evidence"])
               if re.sub(r"\s+", "", m["percent"]) == percent}
    if len(matches) != 1:
        return None
    count, total = matches.pop()
    return {"respondents": count, "total": total}


def make_records(facts, passage, first_number):
    records = []
    for i, fact in enumerate(facts):
        fact = attach_result_assumption(fact, passage["text"])
        response = survey_response(fact)
        bundled_rate = SURVEY_RESPONSE.match(fact["number"]) if response else None
        if bundled_rate:
            fact = {**fact, "number": bundled_rate["percent"].strip()}
        source_text = passage.get("source_text", passage["text"])
        evidence_found, number_found = check_fact(fact, source_text)
        evidence = fact["evidence"]
        if not evidence_found:
            restored = restore_table_evidence(evidence, passage["text"])
            evidence_found = restored != evidence and check_fact({**fact, "evidence": restored}, source_text)[0]
            evidence = restored
        fact_language = likely_language(fact["fact"])
        evidence_language = likely_language(fact["evidence"]) or likely_language(passage["text"])
        conditions = projection_conditions(fact, passage["text"])
        claim_words = f" {words_only(fact['fact'])} "
        measured_claim = fact["fact"]
        for condition in conditions:
            measured_claim = re.sub(re.escape(condition), " ", measured_claim, flags=re.I)
        count = quantity_count(measured_claim, fact["year"], fact["number"])
        multiple = count > 1 or quantity_count(fact["number"]) > 1
        if count and not fact["number"]:
            number_found = False
        records.append({
            "fact_id": f"F{first_number + i:05d}",
            "fact": fact["fact"],
            "number": fact["number"],
            "survey_counts": response,
            "year": fact["year"],
            "quantity_status": fact.get("quantity_status", "unspecified"),
            "scenario_conditions": conditions,
            "scenario_needs_review": any(f" {words_only(c)} " not in claim_words for c in conditions),
            "evidence": evidence,
            "source": passage["citation"],
            "source_file": passage["source_file"],
            "page": passage["page"],
            "passage_id": passage["passage_id"],
            "needs_chart_review": passage.get("needs_chart_review", False),
            "evidence_found": evidence_found,
            "number_found": number_found,
            "multiple_quantities": multiple,
            "language_mismatch": bool(fact_language and evidence_language and
                                      fact_language != evidence_language),
        })
    return records


def quantity_review_indices(records):
    """Indexes of facts whose number needs a check, like a count that does not fit or one evidence used for two years."""
    indices = {i for i, r in enumerate(records) if not r["number_found"] or r.get("multiple_quantities")}
    groups = {}
    endpoints = {}
    for i, r in enumerate(records):
        counts = r.get("survey_counts")
        if counts:
            rate = float(re.match(r"\d+(?:[.,]\d+)?", r["number"])[0].replace(",", "."))
            if (not counts["total"] or counts["respondents"] > counts["total"]
                    or abs(100 * counts["respondents"] / counts["total"] - rate) > 0.5):
                indices.add(i)
        if r["number"] and r["year"]:
            # 'about the same as 2019' does not give the exact value for 2019
            if (r.get("quantity_status") == "observed"
                    and re.search(r"\b(?:ongeveer gelijk aan|roughly unchanged from|about the same as)"
                                  r"\s+(?:het niveau van\s+)?"
                                  + re.escape(r["year"]) + r"\b", r["evidence"], re.I)):
                indices.add(i)
            key = (normalise(r["evidence"]), normalise(r["number"]))
            groups.setdefault(key, []).append(i)
            if r.get("fact"):
                endpoints.setdefault(words_only(r["fact"]), []).append(i)
    for (evidence, _), group in groups.items():
        years = {records[i]["year"] for i in group}
        if len(years) > 1 and any(year not in evidence for year in years):
            indices.update(group)
    for group in endpoints.values():
        if (len({records[i]["year"] for i in group}) > 1
                and len({records[i]["number"] for i in group}) > 1):
            indices.update(group)
    return indices


def select(passages, args):
    if args.passage:
        passages = [p for p in passages if p["passage_id"] in args.passage.split(",")]
    if args.file:
        passages = [p for p in passages if p["source_file"].startswith(args.file)]
    if args.page is not None:
        passages = [p for p in passages if str(p["page"]) == str(args.page)]
    if args.limit:
        passages = passages[:args.limit]
    return passages


def dry_run(passages, show_prompt):
    """No API calls. Count the passages and estimate the tokens (about 4 characters per token)."""
    passages = [prepare_passage(p) for p in passages]
    eligible = [p for p in passages if p["text"] and table_facts(p["text"]) is None]
    chars = sum(len(m["content"]) for p in eligible for m in build_messages(p))
    print(f"{len(passages)} passages, {len(eligible)} initial API calls, up to one correction each")
    print(f"input: about {chars / 4 / 1e6:.2f} million tokens (estimate: 4 characters per token)")
    print("output: depends on the number of facts, roughly 1 to 2 times the passage text")
    if show_prompt and passages:
        for m in build_messages(passages[0]):
            print(f"\n----- {m['role']} -----\n{m['content']}")


def ask_llm(client, passage, tokens, correction=None, indexed=False):
    """extract_facts() that also counts the tokens."""
    facts, usage = extract_facts(client, passage, correction, indexed)
    tokens["in"] += usage.prompt_tokens
    tokens["out"] += usage.completion_tokens
    return facts


def retry_flagged(client, passage, facts, records, number, tokens):
    """Send the flagged facts back to the LLM once and put the answers in their place."""
    quantity_errors = quantity_review_indices(records)
    scenario_errors = {j for j, r in enumerate(records) if r["scenario_needs_review"]}
    evidence_errors = {j for j, r in enumerate(records) if not r["evidence_found"]}
    mismatch = [j for j, r in enumerate(records) if r["language_mismatch"]]

    if not (quantity_errors or scenario_errors or evidence_errors):
        if mismatch:
            return fix_language(client, passage, facts, records, mismatch, number, tokens)
        return records

    targets = sorted(quantity_errors.union(mismatch, scenario_errors, evidence_errors))
    to_fix = [{**facts[j], "original_index": j, "scenario_conditions": records[j]["scenario_conditions"]}
              for j in targets]
    correction = CORRECTION + json.dumps(to_fix, ensure_ascii=False)
    revised = ask_llm(client, passage, tokens, correction, indexed=True)
    if not revised:
        return records

    # One flagged fact can come back as several facts
    replacements = {}
    for candidate in revised:
        index = candidate.pop("original_index")
        if index in targets:
            replacements.setdefault(index, []).append(candidate)
    corrected = [candidate for j, original in enumerate(facts)
                 for candidate in replacements.get(j, [original])]
    return make_records(corrected, passage, number)


def fix_language(client, passage, facts, records, mismatch, number, tokens):
    """Ask the LLM to rewrite facts that are not in the language of their evidence."""
    originals = [facts[j] for j in mismatch]
    correction = LANGUAGE_FIX + json.dumps(originals, ensure_ascii=False)
    revised = ask_llm(client, passage, tokens, correction)
    if len(revised) != len(originals):
        return records

    for j, candidate in zip(mismatch, revised):
        # Only the fact text may change
        if any(candidate[k] != facts[j][k] for k in ("year", "number", "quantity_status", "evidence")):
            continue
        record, = make_records([candidate], passage, number + j)
        if record["language_mismatch"] or not likely_language(candidate["fact"]):
            continue
        if record["number_found"] or not records[j]["number_found"]:
            records[j] = record
    return records


def save_records(records, passage, facts_file, done_file):
    with open(facts_file, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(done_file, "a", encoding="utf-8") as f:
        f.write(passage["passage_id"] + "\n")


def is_flagged(record):
    return (not record["evidence_found"] or record["quantity_needs_review"]
            or record["language_mismatch"] or record["scenario_needs_review"])


def process_passage(client, passage):
    """Extract and check one passage with its own token counts and temporary fact IDs."""
    passage = prepare_passage(passage)
    tokens = {"in": 0, "out": 0}
    records = []
    facts = table_facts(passage["text"])
    if facts is None and passage["text"]:
        facts = ask_llm(client, passage, tokens)
    if facts:
        records = make_records(facts, passage, 1)
        records = retry_flagged(client, passage, facts, records, 1, tokens)
    quantity_errors = quantity_review_indices(records)
    for j, record in enumerate(records):
        record["quantity_needs_review"] = j in quantity_errors
    return passage, records, tokens


def parallel_passages(client, passages, workers):
    """Yield results in input order, keeping at most workers passages in flight.

    Only the caller writes files and assigns final IDs. A failed passage is never
    marked done. Pending work is cancelled if processing or saving fails.
    """
    passages = iter(passages)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        try:
            for _ in range(workers):
                passage = next(passages, None)
                if passage is None:
                    break
                pending.append(pool.submit(process_passage, client, passage))
            while pending:
                yield pending.popleft().result()
                passage = next(passages, None)
                if passage is not None:
                    pending.append(pool.submit(process_passage, client, passage))
        finally:
            for future in pending:
                future.cancel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=config.STEP3_DIR,
                        help="folder for the facts and the list of done passages")
    parser.add_argument("--passage", help="passage id, or several separated by commas")
    parser.add_argument("--file", help="only passages of the file whose name starts with this text")
    parser.add_argument("--page", help="only passages of this page (or sheet name)")
    parser.add_argument("--limit", type=int, help="at most this many passages")
    parser.add_argument("--workers", type=int, default=config.EXTRACTION_WORKERS,
                        help="maximum concurrent passages (default: %(default)s; 1 for sequential)")
    parser.add_argument("--dry-run", action="store_true", help="no API calls, only count and show the prompt")
    parser.add_argument("--yes", action="store_true", help="do not ask before a run of more than 20 passages")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    all_passages = load_passages()
    passages = select(all_passages, args)
    if args.dry_run:
        dry_run(passages, show_prompt=bool(args.passage or args.page is not None))
        return

    if not config.LLM_MODEL:
        sys.exit("Set LLM_MODEL in .env first.")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in .env first.")

    facts_file = args.output_dir / FACTS_FILE.name
    done_file = args.output_dir / DONE_FILE.name
    done = done_passages(done_file)
    todo = [p for p in passages if p["passage_id"] not in done]
    print(f"{len(passages)} passages selected, {len(passages) - len(todo)} already done, {len(todo)} to do")
    if len(todo) > 20 and not args.yes:
        answer = input(f"{len(todo)} passages, up to two API calls each. Go on? (yes/no) ")
        if answer.strip().lower() != "yes":
            return

    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_ids = done.union(p["passage_id"] for p in passages)
    write_chart_review((p for p in all_passages if p["passage_id"] in review_ids), args.output_dir)

    number = next_fact_number(facts_file)
    tokens = {"in": 0, "out": 0}
    print(f"Processing with up to {args.workers} concurrent passages")
    results = parallel_passages(client, todo, args.workers)
    try:
        for i, (passage, records, usage) in enumerate(results, start=1):
            for j, record in enumerate(records):
                record["fact_id"] = f"F{number + j:05d}"
            save_records(records, passage, facts_file, done_file)
            number += len(records)
            for key in tokens:
                tokens[key] += usage[key]

            flagged = sum(is_flagged(r) for r in records)
            print(f"[{i}/{len(todo)}] {passage['passage_id']} {passage['citation']} p{passage['page']}: "
                  f"{len(records)} facts, {flagged} flagged")
    finally:
        results.close()
        client.close()

    print(f"\nTokens used: {tokens['in']} input, {tokens['out']} output")
    print(f"facts -> {facts_file}")


if __name__ == "__main__":
    main()

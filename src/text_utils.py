"""Small text helpers shared by the pipeline scripts."""
import re


def join_lines(prev, line):
    """Join two lines. A word split with a hyphen is joined: 'aan-' + 'vullend' -> 'aanvullend'.

    The hyphen is kept when it is not between two lower case letters ('EU-' + 'landen')
    and before 'en', 'of' and 'tot' ('wet-' + 'en regelgeving').
    """
    if re.search(r"[a-zà-ÿ]-$", prev) and re.match(r"[a-zà-ÿ]", line) \
            and not re.match(r"(?:en|of|tot)\b", line):
        return prev[:-1] + line
    return f"{prev} {line}"

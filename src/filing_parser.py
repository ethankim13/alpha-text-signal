'''
1. Take one filing's raw HTML
2. Extract the Risk Factors and MD&A text

This is a text-processing module:
Input: A string of HTML
Output: Extracted text
'''
from bs4 import BeautifulSoup # used for parsing HTML documents
import re

# Convert raw HTML into plain text that can be pattern-matched
# Strip <tags>, keep visible words
def strip_html(raw_html: str) -> str:
    # Parse raw HTML document
    soup = BeautifulSoup(raw_html, "html.parser")

    # Kill invisible structural elements
    for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
        element.decompose() # removes tag from HTML tree, destroys it

    # Extract text with space separators to avoid word clumping
    plain_text = soup.get_text(separator=" ")

    # Collapse multiple spaces/newlines into single spaces
    clean_text = re.sub(r'\s+', ' ', plain_text).strip() # replace every string with pattern with single space

    return clean_text

SECTION_PATTERNS = {
    "risk_factors": {
        "start": r"Item\s*1A\.?\s*Risk Factors",
        "end": r"Item\s*1B\.|Item\s*2\.",
    },
    "mda": {
        "start": r"Item\s*2\.?\s*Management.s Discussion and Analysis",
        "end": r"Item\s*3\.|Item\s*4\.",
    },
}

# Above section allows section_type (parameter below) to pick start/end pair to use.
def extract_section(plain_text: str, section_type: str) -> str | None:
    """
    Find start of requested section (ex. "Item 1a. Risk Factors")
    and start of the next Item heading after it, then returns everything
    in between. Returns None if the start heading can't be found.
    """
    patterns = SECTION_PATTERNS.get(section_type)
    if patterns is None:
        raise ValueError(f"Unknown section_type: {section_type}")

    # Find every match of the start pattern (not just first).
    # Real section heading is likely the last one (first is likely TOC).
    start_matches = list(re.finditer(patterns["start"], plain_text, flags = re.IGNORECASE)) # finds all non-overlapping matches
    if not start_matches:
        return None
    start_match = start_matches[-1] # last pattern match

    # Start search for end pattern AFTER the start match
    # Prevents early matches from beginning of document (TOC, etc.)
    remaining_text = plain_text[start_match.end():]
    end_match = re.search(patterns["end"], remaining_text, flags = re.IGNORECASE)
    if end_match is None:
        return None

    section_text = remaining_text[:end_match.start()]
    return section_text.strip()







    
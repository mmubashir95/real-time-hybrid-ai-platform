"""Split plain extracted text at the corpus's ``[SECTION-KEY] Title`` lines.

PDF and TXT sources have no markup, so their section headings are recognized
from the bracketed key convention that also appears in HTML/Markdown headings
and in ``sections_index.csv``. Text without markers is returned unchanged with
whatever section was already open.
"""

import re


# A heading occupies a whole line and starts with an upper-case bracketed key,
# e.g. "[DOC-CONTROL] Document ownership...". Inline references such as
# "see [SCOPE]" do not start a line and therefore never open a section.
SECTION_MARKER_PATTERN = re.compile(r"^\[[A-Z0-9][A-Z0-9-]*\] \S")


def split_at_section_markers(
    text: str, open_section: str | None
) -> list[tuple[str | None, str]]:
    """Return ordered ``(section, text)`` segments for one extracted text unit.

    ``open_section`` is the section still in effect from earlier text, such as
    the previous PDF page. Each marker line stays at the start of its segment,
    matching Markdown blocks, so no source text is discarded. Segments that
    contain only whitespace are omitted.
    """
    segments: list[tuple[str | None, str]] = []
    current_section = open_section
    current_lines: list[str] = []

    def append_segment() -> None:
        """Store the accumulated lines under the section that owns them."""
        segment_text = "".join(current_lines).strip()
        if segment_text:
            segments.append((current_section, segment_text))

    for line in text.splitlines(keepends=True):
        if SECTION_MARKER_PATTERN.match(line):
            append_segment()
            current_section = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    append_segment()
    return segments

import re


class CodeChunker:
    """Chunk code by functions/classes, keeping them intact."""

    def chunk(self, text: str, max_tokens: int = 500) -> list[str]:
        if not text.strip():
            return []

        max_chars = max_tokens * 4

        # Try to split by function/class definitions
        # Handles Python, JS/TS, Go, Rust, Java patterns
        patterns = [
            r'\n(?=(?:def |class |function |func |fn |public |private |protected ))',
            r'\n(?=(?:export |const \w+ = |let \w+ = |var \w+ = ))',
        ]

        for pattern in patterns:
            sections = re.split(pattern, text)
            if len(sections) > 1:
                return self._merge_small_chunks(sections, max_chars)

        # Fallback: split by blank lines
        sections = re.split(r'\n\n+', text)
        return self._merge_small_chunks(sections, max_chars)

    def _merge_small_chunks(
        self, sections: list[str], max_chars: int
    ) -> list[str]:
        chunks = []
        current = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current) + len(section) + 2 <= max_chars:
                current = f"{current}\n\n{section}" if current else section
            else:
                if current:
                    chunks.append(current)
                if len(section) > max_chars:
                    # Large function: split at line boundaries
                    lines = section.split("\n")
                    sub = ""
                    for line in lines:
                        if len(sub) + len(line) + 1 <= max_chars:
                            sub = f"{sub}\n{line}" if sub else line
                        else:
                            if sub:
                                chunks.append(sub)
                            sub = line
                    if sub:
                        chunks.append(sub)
                    current = ""
                else:
                    current = section

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

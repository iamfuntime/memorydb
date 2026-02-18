import re


class TranscriptChunker:
    """Chunk transcripts by speaker turns or time windows."""

    def chunk(self, text: str, max_tokens: int = 500) -> list[str]:
        if not text.strip():
            return []

        max_chars = max_tokens * 4

        # Try splitting by speaker labels (e.g., "Speaker 1:", "John:", "[00:01:23]")
        patterns = [
            r'\n(?=\[?\d{1,2}:\d{2})',  # Timestamps
            r'\n(?=\w+:\s)',  # Speaker labels
        ]

        for pattern in patterns:
            sections = re.split(pattern, text)
            if len(sections) > 1:
                return self._merge_turns(sections, max_chars)

        # Fallback: split by paragraphs
        sections = re.split(r'\n\n+', text)
        return self._merge_turns(sections, max_chars)

    def _merge_turns(self, sections: list[str], max_chars: int) -> list[str]:
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
                current = section

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]

import re


class TextChunker:
    """Chunk text by headings, then paragraphs."""

    def chunk(self, text: str, max_tokens: int = 500) -> list[str]:
        if not text.strip():
            return []

        # Rough token estimate: 1 token ~= 4 chars
        max_chars = max_tokens * 4

        # Try splitting by markdown headings first
        sections = re.split(r'\n(?=#{1,3}\s)', text)

        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= max_chars:
                chunks.append(section)
            else:
                # Split long sections by double newlines (paragraphs)
                paragraphs = re.split(r'\n\n+', section)
                current = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if len(current) + len(para) + 2 <= max_chars:
                        current = f"{current}\n\n{para}" if current else para
                    else:
                        if current:
                            chunks.append(current)
                        # If single paragraph is too long, split by sentences
                        if len(para) > max_chars:
                            chunks.extend(self._split_by_sentence(para, max_chars))
                        else:
                            current = para
                            continue
                        current = ""
                if current:
                    chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _split_by_sentence(self, text: str, max_chars: int) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = f"{current} {sent}" if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        return chunks

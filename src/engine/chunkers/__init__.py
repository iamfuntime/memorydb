from typing import Protocol, runtime_checkable


@runtime_checkable
class Chunker(Protocol):
    """Protocol for text chunkers."""

    def chunk(self, text: str, max_tokens: int = 500) -> list[str]:
        """Split text into chunks."""
        ...


def get_chunker(content_type: str) -> Chunker:
    """Get appropriate chunker for content type."""
    if content_type == "code":
        from src.engine.chunkers.code import CodeChunker
        return CodeChunker()
    elif content_type == "audio":
        from src.engine.chunkers.transcript import TranscriptChunker
        return TranscriptChunker()
    else:
        from src.engine.chunkers.text import TextChunker
        return TextChunker()

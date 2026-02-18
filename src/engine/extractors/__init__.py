from typing import Protocol, runtime_checkable


@runtime_checkable
class ContentExtractor(Protocol):
    """Protocol for content extractors."""

    async def extract(self, content: str, **kwargs) -> str:
        """Extract text from content."""
        ...


# Registry of extractors by content type
_EXTRACTORS: dict[str, type] = {}


def register_extractor(content_type: str):
    """Decorator to register an extractor for a content type."""

    def decorator(cls):
        _EXTRACTORS[content_type] = cls
        return cls

    return decorator


def get_extractor(content_type: str) -> ContentExtractor:
    """Get extractor for a content type."""
    if content_type not in _EXTRACTORS:
        raise ValueError(f"No extractor for content type: {content_type}")
    return _EXTRACTORS[content_type]()


# Import all extractors to trigger registration
from src.engine.extractors.text import TextExtractor  # noqa: F401, E402
from src.engine.extractors.url import URLExtractor  # noqa: F401, E402
from src.engine.extractors.pdf import PDFExtractor  # noqa: F401, E402
from src.engine.extractors.image import ImageExtractor  # noqa: F401, E402
from src.engine.extractors.audio import AudioExtractor  # noqa: F401, E402
from src.engine.extractors.code import CodeExtractor  # noqa: F401, E402
from src.engine.extractors.structured import (  # noqa: F401, E402
    CSVExtractor,
    JSONExtractor,
    DOCXExtractor,
    XLSXExtractor,
)

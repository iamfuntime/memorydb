from src.engine.extractors import register_extractor


@register_extractor("text")
class TextExtractor:
    """Pass-through extractor for plain text."""

    async def extract(self, content: str, **kwargs) -> str:
        return content.strip()

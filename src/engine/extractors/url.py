from src.engine.extractors import register_extractor
from src.utils.logger import get_logger
from src.utils.url_validation import SSRFError, safe_fetch, validate_url

logger = get_logger(__name__)


@register_extractor("url")
class URLExtractor:
    """Extract clean text from URLs using trafilatura."""

    async def extract(self, content: str, **kwargs) -> str:
        validate_url(content)

        try:
            import trafilatura

            html = await safe_fetch(content)
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                output_format="txt",
            )

            if not text:
                raise ValueError(f"No text extracted from URL: {content}")

            return text
        except ImportError:
            logger.warning("trafilatura not installed, falling back to httpx")
            return await safe_fetch(content)

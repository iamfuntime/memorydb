from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("url")
class URLExtractor:
    """Extract clean text from URLs using trafilatura."""

    async def extract(self, content: str, **kwargs) -> str:
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(content)
            if downloaded is None:
                raise ValueError(f"Failed to fetch URL: {content}")

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                output_format="txt",
            )

            if not text:
                raise ValueError(f"No text extracted from URL: {content}")

            return text
        except ImportError:
            logger.warning("trafilatura not installed, falling back to httpx")
            import httpx

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(content, timeout=30.0)
                response.raise_for_status()
                return response.text

from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("image")
class ImageExtractor:
    """Extract text from images using OCR."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the image."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(content)
            text = pytesseract.image_to_string(img)
            return text.strip() if text.strip() else "[No text detected in image]"
        except ImportError:
            raise RuntimeError(
                "pytesseract and Pillow are required for image extraction. "
                "Install with: pip install pytesseract Pillow"
            )

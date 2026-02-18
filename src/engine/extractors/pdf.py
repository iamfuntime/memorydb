from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("pdf")
class PDFExtractor:
    """Extract text from PDFs using pymupdf with OCR fallback."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the PDF."""
        try:
            import fitz  # pymupdf

            doc = fitz.open(content)
            pages = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text.strip())
                else:
                    # OCR fallback for scanned pages
                    pages.append(self._ocr_page(page))
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError(
                "pymupdf is required for PDF extraction. "
                "Install with: pip install pymupdf"
            )

    def _ocr_page(self, page) -> str:
        """OCR a page using pytesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img)
        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return ""

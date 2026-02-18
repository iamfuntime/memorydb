from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("csv")
class CSVExtractor:
    """Extract text summary from CSV files."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the CSV."""
        try:
            import pandas as pd

            df = pd.read_csv(content)
            summary = [
                f"CSV with {len(df)} rows and {len(df.columns)} columns.",
                f"Columns: {', '.join(df.columns.tolist())}",
                "",
                "First 10 rows:",
                df.head(10).to_string(index=False),
            ]
            return "\n".join(summary)
        except ImportError:
            raise RuntimeError("pandas is required for CSV extraction")


@register_extractor("json")
class JSONExtractor:
    """Extract text from JSON data."""

    async def extract(self, content: str, **kwargs) -> str:
        """content can be JSON string or file path."""
        import json
        import os

        if os.path.isfile(content):
            with open(content) as f:
                data = json.load(f)
        else:
            data = json.loads(content)

        return json.dumps(data, indent=2, default=str)


@register_extractor("docx")
class DOCXExtractor:
    """Extract text from DOCX files."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the DOCX."""
        try:
            from docx import Document

            doc = Document(content)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise RuntimeError("python-docx is required for DOCX extraction")


@register_extractor("xlsx")
class XLSXExtractor:
    """Extract text summary from XLSX files."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the XLSX."""
        try:
            import pandas as pd

            sheets = pd.read_excel(content, sheet_name=None)
            parts = []
            for name, df in sheets.items():
                parts.append(f"Sheet: {name} ({len(df)} rows, {len(df.columns)} cols)")
                parts.append(f"Columns: {', '.join(df.columns.astype(str).tolist())}")
                parts.append(df.head(10).to_string(index=False))
                parts.append("")
            return "\n".join(parts)
        except ImportError:
            raise RuntimeError("pandas and openpyxl are required for XLSX extraction")

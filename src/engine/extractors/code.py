from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("code")
class CodeExtractor:
    """Extract code with language detection and metadata."""

    async def extract(self, content: str, **kwargs) -> str:
        """content is the raw code string or file path."""
        import os

        # If it's a file path, read it
        if os.path.isfile(content):
            with open(content) as f:
                code = f.read()
            filename = os.path.basename(content)
            lang = self._detect_language(filename)
            return f"Language: {lang}\nFile: {filename}\n\n{code}"

        # Otherwise treat as raw code
        return content

    def _detect_language(self, filename: str) -> str:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".cpp": "cpp",
            ".c": "c",
            ".sh": "bash",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
        }
        for ext, lang in ext_map.items():
            if filename.endswith(ext):
                return lang
        return "unknown"

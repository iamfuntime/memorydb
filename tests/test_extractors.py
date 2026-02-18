import pytest

from src.engine.extractors import get_extractor


@pytest.mark.asyncio
async def test_text_extractor():
    extractor = get_extractor("text")
    result = await extractor.extract("  Hello World  ")
    assert result == "Hello World"


@pytest.mark.asyncio
async def test_code_extractor_raw():
    extractor = get_extractor("code")
    result = await extractor.extract("def hello():\n    return 'hi'")
    assert "def hello" in result


def test_get_unknown_extractor():
    with pytest.raises(ValueError, match="No extractor for content type"):
        get_extractor("nonexistent_type")


def test_all_extractors_registered():
    """Verify all expected content types have extractors."""
    expected = ["text", "url", "pdf", "image", "audio", "code", "csv", "json", "docx", "xlsx"]
    for ct in expected:
        extractor = get_extractor(ct)
        assert extractor is not None

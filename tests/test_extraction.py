import pytest

from src.engine.extraction import MemoryExtractor


class FakeLLM:
    """Fake LLM provider that returns canned responses."""

    def __init__(self, response: str):
        self._response = response

    async def complete(self, prompt: str, system: str = "") -> str:
        return self._response


@pytest.mark.asyncio
async def test_extract_basic():
    llm = FakeLLM('{"memories": [{"content": "User likes Python", "type": "preference", "confidence": 0.9, "tags": ["python"]}]}')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("I really like Python for backend work.")
    assert len(result) == 1
    assert result[0]["content"] == "User likes Python"
    assert result[0]["type"] == "preference"
    assert result[0]["confidence"] == 0.9
    assert "python" in result[0]["tags"]


@pytest.mark.asyncio
async def test_extract_multiple_memories():
    llm = FakeLLM('{"memories": [{"content": "Fact one", "type": "fact"}, {"content": "Fact two", "type": "insight", "confidence": 0.7}]}')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("Some text with multiple facts.")
    assert len(result) == 2
    assert result[0]["type"] == "fact"
    assert result[1]["type"] == "insight"
    assert result[1]["confidence"] == 0.7


@pytest.mark.asyncio
async def test_extract_empty_input():
    llm = FakeLLM("should not be called")
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("")
    assert result == []


@pytest.mark.asyncio
async def test_extract_empty_whitespace():
    llm = FakeLLM("should not be called")
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("   \n  ")
    assert result == []


@pytest.mark.asyncio
async def test_extract_json_in_code_block():
    llm = FakeLLM('```json\n{"memories": [{"content": "Uses Docker", "type": "fact"}]}\n```')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("We deploy with Docker.")
    assert len(result) == 1
    assert result[0]["content"] == "Uses Docker"


@pytest.mark.asyncio
async def test_extract_malformed_json():
    llm = FakeLLM("This is not valid JSON at all")
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("Some text.")
    assert result == []


@pytest.mark.asyncio
async def test_extract_missing_content_field():
    llm = FakeLLM('{"memories": [{"type": "fact"}, {"content": "Valid", "type": "fact"}]}')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("Some text.")
    assert len(result) == 1
    assert result[0]["content"] == "Valid"


@pytest.mark.asyncio
async def test_extract_confidence_clamped():
    llm = FakeLLM('{"memories": [{"content": "Over confident", "type": "fact", "confidence": 5.0}]}')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("Some text.")
    assert result[0]["confidence"] == 1.0


@pytest.mark.asyncio
async def test_extract_defaults():
    """Missing type and confidence should get defaults."""
    llm = FakeLLM('{"memories": [{"content": "Just content"}]}')
    extractor = MemoryExtractor(llm)
    result = await extractor.extract("Some text.")
    assert result[0]["type"] == "fact"
    assert result[0]["confidence"] == 0.8
    assert result[0]["tags"] == []


@pytest.mark.asyncio
async def test_extract_truncates_long_text():
    llm = FakeLLM('{"memories": [{"content": "Summary", "type": "fact"}]}')
    extractor = MemoryExtractor(llm)
    long_text = "x" * 10000
    result = await extractor.extract(long_text)
    assert len(result) == 1

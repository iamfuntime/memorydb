from src.models.document import DocumentCreate
from src.models.memory import MemoryType
from src.models.search import SearchRequest


def test_document_create_text():
    doc = DocumentCreate(content="hello", container="test", content_type="text")
    assert doc.content == "hello"
    assert doc.tags == []


def test_document_create_url():
    doc = DocumentCreate(
        content="https://example.com", container="test", content_type="url"
    )
    assert doc.content_type == "url"


def test_memory_types():
    assert MemoryType.FACT == "fact"
    assert MemoryType.PREFERENCE == "preference"
    assert MemoryType.EPISODE == "episode"


def test_search_request_defaults():
    req = SearchRequest(query="test")
    assert req.limit == 10
    assert req.include_related is False

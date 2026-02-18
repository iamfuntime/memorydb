from src.engine.chunkers.text import TextChunker
from src.engine.chunkers.code import CodeChunker


def test_text_chunker_short():
    chunker = TextChunker()
    chunks = chunker.chunk("Hello world")
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_text_chunker_headings():
    text = "# Section 1\nContent one.\n\n# Section 2\nContent two."
    chunker = TextChunker()
    chunks = chunker.chunk(text)
    assert len(chunks) == 2


def test_text_chunker_long_paragraph():
    # Create a long text that exceeds max_tokens
    long_text = "This is a sentence. " * 200  # ~800 tokens
    chunker = TextChunker()
    chunks = chunker.chunk(long_text, max_tokens=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100 * 4 + 100  # Allow some tolerance


def test_text_chunker_empty():
    chunker = TextChunker()
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_code_chunker_functions():
    code = '''def foo():
    return 1

def bar():
    return 2

def baz():
    return 3'''
    chunker = CodeChunker()
    chunks = chunker.chunk(code, max_tokens=50)
    assert len(chunks) >= 1


def test_code_chunker_empty():
    chunker = CodeChunker()
    assert chunker.chunk("") == []

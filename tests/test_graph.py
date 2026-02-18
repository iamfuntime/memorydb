import json

from src.engine.graph import GraphBuilder


def test_parse_response_valid():
    builder = GraphBuilder.__new__(GraphBuilder)
    response = json.dumps({
        "relationships": [
            {
                "existing_memory_id": "abc-123",
                "type": "updates",
                "confidence": 0.9,
                "reason": "supersedes old info",
            }
        ]
    })
    result = builder._parse_response(response)
    assert len(result) == 1
    assert result[0]["type"] == "updates"
    assert result[0]["existing_memory_id"] == "abc-123"


def test_parse_response_code_block():
    builder = GraphBuilder.__new__(GraphBuilder)
    response = '```json\n{"relationships": [{"existing_memory_id": "x", "type": "extends", "confidence": 0.8}]}\n```'
    result = builder._parse_response(response)
    assert len(result) == 1
    assert result[0]["type"] == "extends"


def test_parse_response_empty_relationships():
    builder = GraphBuilder.__new__(GraphBuilder)
    response = '{"relationships": []}'
    result = builder._parse_response(response)
    assert result == []


def test_parse_response_invalid_json():
    builder = GraphBuilder.__new__(GraphBuilder)
    result = builder._parse_response("not json")
    assert result == []


def test_parse_response_plain_code_block():
    builder = GraphBuilder.__new__(GraphBuilder)
    response = '```\n{"relationships": [{"existing_memory_id": "y", "type": "derives", "confidence": 0.7}]}\n```'
    result = builder._parse_response(response)
    assert len(result) == 1
    assert result[0]["type"] == "derives"

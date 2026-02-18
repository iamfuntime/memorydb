import json
from typing import Optional

from src.engine.providers.llm import LLMProvider
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Your job is to extract discrete, meaningful memories from text content.

For each piece of content, extract individual memories and classify them:
- fact: Objective information (e.g., "Team uses PostgreSQL")
- preference: Preferences or opinions (e.g., "User prefers dark mode")
- episode: Events that happened (e.g., "Had meeting on Feb 18")
- insight: Patterns or learnings (e.g., "User asks about Docker frequently")
- task: Action items (e.g., "Need to review PR #42")

Respond ONLY with valid JSON in this format:
{
  "memories": [
    {
      "content": "The extracted memory as a clear, standalone statement",
      "type": "fact|preference|episode|insight|task",
      "confidence": 0.0-1.0,
      "tags": ["relevant", "tags"]
    }
  ]
}

Rules:
- Each memory should be a clear, standalone statement
- Remove filler words and noise
- Merge duplicate information
- Be concise but preserve meaning
- Confidence reflects how certain the information is
- Only extract meaningful information, skip greetings and filler"""


class MemoryExtractor:
    """Extract structured memories from text using an LLM."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def extract(self, text: str) -> list[dict]:
        """Extract memories from text content."""
        if not text.strip():
            return []

        # Truncate very long texts to avoid token limits
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Truncated]"

        prompt = f"Extract memories from the following content:\n\n{text}"

        try:
            response = await self.llm.complete(prompt, EXTRACTION_SYSTEM_PROMPT)
            return self._parse_response(response)
        except Exception as e:
            logger.error("extraction.failed", error=str(e))
            return []

    def _parse_response(self, response: str) -> list[dict]:
        """Parse LLM response into structured memories."""
        try:
            # Try to find JSON in the response
            text = response.strip()

            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            if isinstance(data, dict) and "memories" in data:
                memories = data["memories"]
            elif isinstance(data, list):
                memories = data
            else:
                return []

            # Validate each memory
            valid = []
            for m in memories:
                if not isinstance(m, dict):
                    continue
                if "content" not in m:
                    continue

                valid.append({
                    "content": str(m["content"]).strip(),
                    "type": m.get("type", "fact"),
                    "confidence": min(1.0, max(0.0, float(m.get("confidence", 0.8)))),
                    "tags": m.get("tags", []),
                })

            return valid

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("extraction.parse_failed", error=str(e))
            return []

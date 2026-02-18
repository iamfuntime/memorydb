from src.engine.extractors import register_extractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@register_extractor("audio")
class AudioExtractor:
    """Transcribe audio files using OpenAI Whisper API."""

    async def extract(self, content: str, **kwargs) -> str:
        """content should be a file path to the audio file."""
        api_key = kwargs.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key required for audio transcription")

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)

        with open(content, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcript.text

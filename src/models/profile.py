from pydantic import BaseModel


class ProfileResponse(BaseModel):
    container: str
    static_facts: list[str] = []
    recent_context: list[str] = []
    relevant_memories: list[dict] = []

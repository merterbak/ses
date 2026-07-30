from pydantic import BaseModel, ConfigDict


class SpeechRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model: str = ""
    input: str
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    lang: str | None = None
    stream: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="ignore")

    model: str = ""
    messages: list[dict] = []
    stream: bool = False
    think: bool = True
    conversation: str | None = None
    prompt: str | None = None
    system: str | None = None


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str | None = None
    brain: str | None = None


class ConversationRename(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str

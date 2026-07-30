import json
from fastapi.responses import StreamingResponse
from ... import SesError
from ...core import conversations, llm, paths
from ...core.llm import THINKING
from ..schemas import ChatRequest, ConversationCreate, ConversationRename

DEFAULT_SYSTEM = "You are a helpful assistant. Answer clearly and conversationally."


def register(app, _context):
    @app.get("/api/brain")
    def brain_info():
        brain = llm.detect()
        if brain is None:
            return {"available": False, "hint": llm.NO_BRAIN_HINT}

        try:
            models = brain.models()
        except Exception:
            models = []
        preferred = paths.default_for("brain")
        return {
            "available": True,
            "backend": brain.label,
            "kind": brain.kind,
            "base": brain.base_url,
            "models": models,
            "default": preferred if preferred in models else (models[0] if models else None),
            "loaded": brain.loaded_models(),
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        brain = llm.detect()
        if brain is None:
            raise SesError(llm.NO_BRAIN_HINT)
        model = brain.resolve_model(request.model or paths.default_for("brain"))

        conversation, messages = build_messages(request)
        if not messages:
            raise SesError("no messages to send")

        def remember(reply):
            if conversation is not None:
                conversations.append(conversation, "assistant", reply)

        if request.stream:
            return StreamingResponse(
                stream(brain, model, messages, conversation, remember, request.think),
                media_type="application/x-ndjson",
            )

        try:
            reply = brain.chat(model, messages)
        except SesError:
            raise
        except Exception as error:
            raise SesError(f"{brain.label} chat failed: {error}") from error

        remember(reply)
        return {
            "model": model,
            "backend": brain.label,
            "conversation": conversation["id"] if conversation else None,
            "message": {"role": "assistant", "content": reply},
        }

    @app.get("/api/conversations")
    def list_conversations():
        return {"conversations": conversations.list_all()}

    @app.post("/api/conversations")
    def create_conversation(request: ConversationCreate):
        return conversations.create(brain=request.brain, title=request.title)

    @app.get("/api/conversations/{conversation_id}")
    def get_conversation(conversation_id: str):
        conversation = conversations.load(conversation_id)
        if conversation is None:
            raise SesError(f"conversation '{conversation_id}' not found")
        return conversation

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(conversation_id: str, request: ConversationRename):
        conversation = conversations.rename(conversation_id, request.title)
        if conversation is None:
            raise SesError(f"conversation '{conversation_id}' not found")
        return conversations.summary(conversation)

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        return {"deleted": conversations.delete(conversation_id)}


def build_messages(request):
    system = request.system or DEFAULT_SYSTEM

    if request.conversation is not None:
        conversation = conversations.load(request.conversation)
        if conversation is None:
            raise SesError(f"conversation '{request.conversation}' not found")
        if request.prompt:
            conversations.append(conversation, "user", request.prompt)
        history = [
            {"role": message["role"], "content": message["content"]}
            for message in conversation["messages"]
        ]
        return conversation, [{"role": "system", "content": system}] + history

    if request.prompt is not None:
        return None, [
            {"role": "system", "content": system},
            {"role": "user", "content": request.prompt},
        ]

    return None, request.messages


def stream(brain, model, messages, conversation, remember, think=True):
    answer = ""
    try:
        for channel, token in brain.chat_stream(model, messages, think=think):
            if channel == THINKING:
                yield json.dumps({"model": model, "thinking": token}) + "\n"
            else:
                answer += token
                yield json.dumps({"model": model, "token": token}) + "\n"

        remember(answer)
        yield json.dumps(
            {
                "model": model,
                "backend": brain.label,
                "done": True,
                "conversation": conversation["id"] if conversation else None,
                "message": {"role": "assistant", "content": answer},
            }
        ) + "\n"
    except Exception as error:
        yield json.dumps({"error": str(error)}) + "\n"

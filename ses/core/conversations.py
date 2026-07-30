import uuid
from datetime import datetime, timezone
from . import paths

MAX_TITLE_LENGTH = 48
NEW_CHAT_TITLE = "New chat"
FILE_PREFIX = "conv_"


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_for(conversation_id):
    return paths.conversations_dir() / f"{conversation_id}.json"


def title_from(text):
    condensed = " ".join(text.strip().split())
    if not condensed:
        return NEW_CHAT_TITLE
    if len(condensed) > MAX_TITLE_LENGTH:
        return condensed[:MAX_TITLE_LENGTH] + "…"
    return condensed


def create(brain=None, title=None):
    conversation = {
        "id": FILE_PREFIX + uuid.uuid4().hex[:12],
        "title": title or NEW_CHAT_TITLE,
        "brain": brain,
        "created": timestamp(),
        "updated": timestamp(),
        "messages": [],
    }
    save(conversation)
    return conversation


def save(conversation):
    conversation["updated"] = timestamp()
    paths.write_json(file_for(conversation["id"]), conversation)


def load(conversation_id):
    return paths.read_json(file_for(conversation_id))


def delete(conversation_id):
    path = file_for(conversation_id)
    if path.is_file():
        path.unlink()
        return True
    return False


def rename(conversation_id, title):
    conversation = load(conversation_id)
    if conversation is None:
        return None
    conversation["title"] = title.strip()[:MAX_TITLE_LENGTH] or NEW_CHAT_TITLE
    save(conversation)
    return conversation


def append(conversation, role, content):
    conversation["messages"].append({"role": role, "content": content})
    if role == "user" and conversation.get("title") in (None, "", NEW_CHAT_TITLE):
        conversation["title"] = title_from(content)
    save(conversation)
    return conversation


def summary(conversation):
    return {
        "id": conversation["id"],
        "title": conversation.get("title", NEW_CHAT_TITLE),
        "brain": conversation.get("brain"),
        "updated": conversation.get("updated", ""),
        "created": conversation.get("created", ""),
        "messages": len(conversation.get("messages", [])),
    }


def list_all():
    summaries = []
    for path in paths.conversations_dir().glob(f"{FILE_PREFIX}*.json"):
        conversation = paths.read_json(path)
        if conversation:
            summaries.append(summary(conversation))
    summaries.sort(key=lambda item: item["updated"], reverse=True)
    return summaries


def latest_id():
    conversations = list_all()
    return conversations[0]["id"] if conversations else None

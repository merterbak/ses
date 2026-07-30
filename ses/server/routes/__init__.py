from . import chat, models, speech, transcription

MODULES = (models, speech, transcription, chat)


def register_all(app, context):
    for module in MODULES:
        module.register(app, context)

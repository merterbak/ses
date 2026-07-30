from . import assistant, models, serve, speech

MODULES = (models, speech, assistant, serve)


def register_all(app):
    for module in MODULES:
        module.register(app)

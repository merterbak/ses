import numpy as np
from .. import SesError
from ..audio import WHISPER_SAMPLE_RATE

MAX_NEW_TOKENS = 440


class TransformersSttEngine:
    kind = "stt"

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        directory = str(model_dir)
        self.processor = AutoProcessor.from_pretrained(directory)
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(directory)
            self.generates = True
        except (ValueError, OSError):
            from transformers import AutoModelForCTC

            self.model = AutoModelForCTC.from_pretrained(directory)
            self.generates = False
        self.model.eval()

    def transcribe(
        self,
        samples,
        language=None,
        task="transcribe",
        word_timestamps=False,
        temperature=None,
        initial_prompt=None,
    ):
        audio = np.asarray(samples, dtype=np.float32)
        inputs = self.processor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        )
        try:
            with self.torch.no_grad():
                text = self.decode(inputs, language, task)
        except Exception as error:
            raise SesError(f"transformers failed to transcribe: {error}") from error

        return {"text": text.strip(), "segments": [], "language": language}

    def decode(self, inputs, language, task):
        if not self.generates:
            logits = self.model(**inputs).logits
            ids = self.torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(ids)[0]

        options = {"max_new_tokens": MAX_NEW_TOKENS}
        if language:
            options["language"] = language
        if task == "translate":
            options["task"] = "translate"
        try:
            ids = self.model.generate(**inputs, **options)
        except (TypeError, ValueError):
            ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


class TransformersTtsEngine:
    kind = "tts"

    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        directory = str(model_dir)
        self.model = VitsModel.from_pretrained(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model.eval()
        self.rate = int(self.model.config.sampling_rate)

    def voices(self):
        speakers = getattr(self.model.config, "num_speakers", 1) or 1
        if speakers > 1:
            return [str(index) for index in range(speakers)]
        return ["default"]

    def synth(self, text, voice=None, speed=1.0, lang=None):
        text = (text or "").strip()
        if not text:
            raise SesError("nothing to say, input text is empty")

        original = getattr(self.model, "speaking_rate", None)
        if original is not None and speed:
            self.model.speaking_rate = original * speed

        inputs = self.tokenizer(text, return_tensors="pt")
        try:
            with self.torch.no_grad():
                waveform = self.model(**inputs).waveform[0].cpu().numpy()
        except Exception as error:
            raise SesError(f"transformers failed to synthesize: {error}") from error
        finally:
            if original is not None:
                self.model.speaking_rate = original

        return np.asarray(waveform, dtype=np.float32), self.rate

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import httpx

from .embeddings import FeatureHashEmbedder, mean_pool
from .models import OpenRecord, Patch


class ClapAudioEmbedderError(RuntimeError):
    pass


class ClapAudioEmbedder(FeatureHashEmbedder):
    """Audio-native CLAP provider for speech, music, and environmental sound retrieval."""

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        dimension: int = 512,
        max_audio_s: float = 30.0,
        timeout_s: float = 20.0,
    ):
        super().__init__(dimension=dimension)
        self.backend = "clap"
        self.model_name = model_name
        self.max_audio_s = float(max_audio_s)
        self.timeout_s = float(timeout_s)
        self._model = None
        self._processor = None
        self._http = httpx.Client(timeout=self.timeout_s, follow_redirects=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except Exception as exc:
            raise ClapAudioEmbedderError("CLAP dependencies are missing. Install with `uv sync --extra audio`.") from exc

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        try:
            self._processor = ClapProcessor.from_pretrained(self.model_name)
            self._model = ClapModel.from_pretrained(self.model_name, torch_dtype=dtype).to(device)
            self._model.eval()
        except Exception as exc:
            raise ClapAudioEmbedderError(f"Failed to load {self.model_name}: {exc}") from exc

    @property
    def _device(self):
        self._load_model()
        return next(self._model.parameters()).device

    def embed_text(self, text: str) -> list[float]:
        self._load_model()
        import torch
        import torch.nn.functional as F

        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = self._model.get_text_features(**inputs)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings[0].detach().cpu().float().tolist()[: self.dimension]

    def embed_record_audio(self, record: OpenRecord, patches: list[Patch], evidence_text: str) -> list[float]:
        vectors: list[list[float]] = []
        for value in _audio_sources(record, patches):
            try:
                vectors.append(self.embed_audio(value))
            except Exception:
                continue
        if evidence_text:
            vectors.append(self.embed_text(evidence_text))
        return mean_pool(vectors, self.dimension)

    def embed_audio(self, value: str) -> list[float]:
        self._load_model()
        import torch
        import torch.nn.functional as F

        waveform, sampling_rate = self._load_audio(value)
        inputs = self._processor(audios=waveform, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = self._model.get_audio_features(**inputs)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings[0].detach().cpu().float().tolist()[: self.dimension]

    def _load_audio(self, value: str) -> tuple[Any, int]:
        try:
            import librosa
        except Exception as exc:
            raise ClapAudioEmbedderError("Audio decoding needs `uv sync --extra audio`.") from exc

        value = str(value)
        sampling_rate = 48_000
        if value.startswith(("http://", "https://")):
            response = self._http.get(value)
            response.raise_for_status()
            suffix = Path(value).suffix or ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
                handle.write(response.content)
                handle.flush()
                waveform, loaded_rate = librosa.load(
                    handle.name,
                    sr=sampling_rate,
                    mono=True,
                    duration=self.max_audio_s,
                )
            return waveform, int(loaded_rate)

        path = Path(value.removeprefix("file://")).expanduser()
        waveform, loaded_rate = librosa.load(path, sr=sampling_rate, mono=True, duration=self.max_audio_s)
        return waveform, int(loaded_rate)


def _audio_sources(record: OpenRecord, patches: list[Patch]) -> list[str]:
    sources: list[str] = []
    for patch in patches:
        if patch.kind.startswith("audio") and (patch.source_file or patch.asset_url):
            sources.append(patch.source_file or patch.asset_url)
    for asset in record.assets:
        if asset.url and (asset.kind.startswith("audio") or "audio" in asset.mime_type):
            sources.append(asset.url)
    seen: set[str] = set()
    unique: list[str] = []
    for source in sources:
        if source and source not in seen:
            unique.append(source)
            seen.add(source)
    return unique

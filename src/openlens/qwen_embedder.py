from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from .embeddings import FeatureHashEmbedder, mean_pool
from .models import OpenRecord, Patch


MODEL_ALIASES = {
    "qwen2b": "Qwen/Qwen3-VL-Embedding-2B",
    "qwen8b": "Qwen/Qwen3-VL-Embedding-8B",
}


class QwenEmbedderError(RuntimeError):
    pass


class QwenMultimodalEmbedder(FeatureHashEmbedder):
    """Qwen multimodal embedding provider.

    This uses the Qwen3-VL-Embedding model family, which is the Qwen retrieval
    model line for text, image, screenshot, video, and mixed-modal inputs. A
    Qwen3.5-compatible local model path can be supplied through
    `OPENLENS_QWEN_MODEL` if its processor exposes the same Qwen3-VL interface.
    """

    def __init__(
        self,
        model_name: str = "qwen2b",
        dimension: int = 768,
        batch_size: int = 1,
        max_frames: int = 32,
        fps: float = 1.0,
    ):
        super().__init__(dimension=dimension)
        self.backend = "qwen"
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.dimension = int(dimension)
        self.batch_size = max(1, int(batch_size))
        self.max_frames = int(max_frames)
        self.fps = float(fps)
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn.functional as F  # noqa: F401
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLConfig, Qwen3VLModel, Qwen3VLPreTrainedModel
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        except ImportError as exc:
            raise QwenEmbedderError(
                "Qwen multimodal embedding dependencies are missing. "
                "Install with `uv sync --extra qwen`."
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
            os.environ.setdefault("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
        else:
            device = "cpu"
            dtype = torch.float32

        class _Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
            config: Qwen3VLConfig

            def __init__(self, config: Qwen3VLConfig):
                super().__init__(config)
                self.model = Qwen3VLModel(config)
                self.post_init()

            def get_input_embeddings(self):
                return self.model.get_input_embeddings()

            def set_input_embeddings(self, value):
                self.model.set_input_embeddings(value)

            def forward(self, **kwargs: Any):
                return self.model(**kwargs)

        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
        if device == "mps":
            load_kwargs["attn_implementation"] = "eager"
        try:
            self._processor = Qwen3VLProcessor.from_pretrained(self.model_name, padding_side="right")
            self._model = _Qwen3VLForEmbedding.from_pretrained(self.model_name, **load_kwargs).to(device)
            self._model.eval()
        except Exception as exc:
            raise QwenEmbedderError(f"Failed to load {self.model_name}: {exc}") from exc

    @staticmethod
    def _pool_last(hidden_state, attention_mask):
        import torch

        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _encode_objects(self, objects: list[dict[str, Any]], instruction: str) -> list[list[float]]:
        results: list[list[float]] = []
        for start in range(0, len(objects), self.batch_size):
            results.extend(self._encode_batch(objects[start : start + self.batch_size], instruction))
        return results

    def _encode_batch(self, objects: list[dict[str, Any]], instruction: str) -> list[list[float]]:
        self._load_model()
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": _object_to_content(obj, fps=self.fps, max_frames=self.max_frames)},
            ]
            for obj in objects
        ]
        prompts = [
            self._processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        all_images = []
        all_videos = []
        all_video_metadata = []
        video_kwargs: dict[str, Any] = {}
        for conv in conversations:
            images, video_inputs, kwargs = process_vision_info(
                conv,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if images:
                all_images.extend(images)
            if video_inputs:
                videos, metadata = zip(*video_inputs)
                all_videos.extend(videos)
                all_video_metadata.extend(metadata)
            video_kwargs.update(kwargs)

        inputs = self._processor(
            text=prompts,
            images=all_images or None,
            videos=all_videos or None,
            video_metadata=all_video_metadata or None,
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        t0 = time.monotonic()
        with torch.no_grad():
            outputs = self._model(**inputs)
            embeddings = self._pool_last(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        if os.getenv("OPENLENS_QWEN_VERBOSE"):
            print(f"qwen encoded {len(objects)} objects in {time.monotonic() - t0:.2f}s", file=sys.stderr)
        return [self._truncate(vec) for vec in embeddings]

    def _truncate(self, embedding) -> list[float]:
        import torch

        vec = embedding[: self.dimension]
        norm = torch.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.cpu().float().tolist()

    def embed_text(self, text: str) -> list[float]:
        return self._encode_objects([{"text": text}], "Represent the user's query for multimodal retrieval.")[0]

    def embed_record(self, record: OpenRecord) -> list[float]:
        patches = self.patch_record(record, max_patches=1)
        return mean_pool(self.embed_patches(patches), self.dimension)

    def embed_patches(self, patches: list[Patch]) -> list[list[float]]:
        objects = []
        for patch in patches:
            if patch.kind.startswith("video") and (patch.source_file or patch.asset_url):
                objects.append({"video": patch.source_file or patch.asset_url, "text": patch.text})
            elif (
                patch.kind.startswith("audio")
                and (patch.source_file or patch.asset_url)
                and os.getenv("OPENLENS_QWEN_RAW_AUDIO") == "1"
            ):
                objects.append({"audio": patch.source_file or patch.asset_url, "text": patch.text})
            elif patch.kind.startswith("visual") and patch.asset_url:
                objects.append({"image": patch.asset_url, "text": patch.text})
            else:
                objects.append({"text": patch.text})
        return self._encode_objects(objects, "Represent this document patch for multimodal retrieval.")

    def embed_query_patches(self, query: str, max_patches: int = 8) -> list[list[float]]:
        return self._encode_objects([{"text": query}], "Represent the user's query for multimodal retrieval.")


def _object_to_content(obj: dict[str, Any], fps: float, max_frames: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if obj.get("text"):
        content.append({"type": "text", "text": str(obj["text"])})
    if obj.get("image"):
        content.append({"type": "image", "image": _media_path(obj["image"])})
    if obj.get("video"):
        content.append({"type": "video", "video": _media_path(obj["video"]), "fps": fps, "max_frames": max_frames})
    if obj.get("audio"):
        content.append({"type": "audio", "audio": _media_path(obj["audio"])})
    if not content:
        content.append({"type": "text", "text": ""})
    return content


def _media_path(value: str) -> str:
    value = str(value)
    if value.startswith(("http://", "https://", "file://")):
        return value
    path = Path(value).expanduser()
    return "file://" + str(path.resolve())


def make_embedder(
    backend: str,
    dimension: int,
    model_name: str = "",
    batch_size: int = 1,
    max_frames: int = 32,
    fps: float = 1.0,
) -> FeatureHashEmbedder:
    if backend == "qwen":
        return QwenMultimodalEmbedder(
            model_name=model_name or "qwen8b",
            dimension=dimension,
            batch_size=batch_size,
            max_frames=max_frames,
            fps=fps,
        )
    return FeatureHashEmbedder(dimension=dimension)

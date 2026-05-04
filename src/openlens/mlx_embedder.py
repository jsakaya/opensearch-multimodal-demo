from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .embeddings import FeatureHashEmbedder, mean_pool, normalize
from .models import OpenRecord, Patch
from .text import compose_search_text


DEFAULT_MLX_TEXT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
DEFAULT_MLX_QWEN_VL_MODEL = "mlx-community/Qwen3-VL-Embedding-2B-6bit"
DEFAULT_MLX_COLQWEN_MODEL = "qnguyen3/colqwen2_5-v0.2-mlx-4bit"


class MlxEmbedderError(RuntimeError):
    pass


def mlx_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {"available": False}
    try:
        import mlx.core as mx

        status.update({"available": True, "default_device": str(mx.default_device())})
    except Exception as exc:
        status["detail"] = f"{type(exc).__name__}: {exc}"
        return status

    try:
        import mlx_embeddings

        status["mlx_embeddings_available"] = True
        status["mlx_embeddings_path"] = getattr(mlx_embeddings, "__file__", "")
    except Exception as exc:
        status["mlx_embeddings_available"] = False
        status["mlx_embeddings_detail"] = f"{type(exc).__name__}: {exc}"
    return status


class MlxTextEmbedder(FeatureHashEmbedder):
    """MLX text embedding provider for Apple Silicon trial indexing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_TEXT_MODEL,
        dimension: int = 1024,
        max_length: int = 512,
    ):
        super().__init__(dimension=dimension)
        self.backend = "mlx"
        self.model_name = model_name
        self.max_length = int(max_length)
        self._model = None
        self._processor = None
        self._generate = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_embeddings import generate, load
        except Exception as exc:
            raise MlxEmbedderError(
                "MLX embedding dependencies are missing or failed to import. "
                "Run with Python 3.12 and `uv sync --extra mlx`."
            ) from exc
        try:
            self._model, self._processor = load(self.model_name)
            self._generate = generate
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to load MLX model {self.model_name}: {exc}") from exc

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        try:
            out = self._generate(  # type: ignore[misc]
                self._model,
                self._processor,
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
            )
            embeddings = getattr(out, "text_embeds", out)
            return [_fit_dimension(row, self.dimension) for row in _as_numpy_rows(embeddings)]
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to encode text with {self.model_name}: {exc}") from exc

    def embed_record(self, record: OpenRecord) -> list[float]:
        prefix = f"modality:{record.modality} source:{record.source} title:{record.title} tags:{' '.join(record.tags)}"
        return self.embed_text(prefix + " " + compose_search_text(record))


class MlxQwenVlEmbedder(FeatureHashEmbedder):
    """Qwen3-VL MLX smoke/trial provider for image-text embeddings on Apple GPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_QWEN_VL_MODEL,
        dimension: int = 2048,
        max_length: int = 512,
    ):
        super().__init__(dimension=dimension)
        self.backend = "mlx-qwen-vl"
        self.model_name = model_name
        self.max_length = int(max_length)
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_embeddings import load
        except Exception as exc:
            raise MlxEmbedderError(
                "Qwen3-VL MLX dependencies are missing. Run with Python 3.12 and `uv sync --extra mlx`."
            ) from exc
        try:
            self._model, self._processor = load(self.model_name)
            self._patch_qwen_processor()
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to load MLX Qwen3-VL model {self.model_name}: {exc}") from exc

    def _patch_qwen_processor(self) -> None:
        processor = getattr(self._processor, "processor", self._processor)
        if not hasattr(processor, "image_ids") and hasattr(processor, "image_token_id"):
            processor.image_ids = [processor.image_token_id]
        if not hasattr(processor, "video_ids") and hasattr(processor, "video_token_id"):
            processor.video_ids = [processor.video_token_id]
        if not hasattr(processor, "audio_ids"):
            processor.audio_ids = []

    def embed_text(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_record(self, record: OpenRecord) -> list[float]:
        patches = self.patch_record(record, max_patches=1)
        return self.embed_patches(patches)[0] if patches else self.embed_text(compose_search_text(record))

    def embed_patches(self, patches: list[Patch]) -> list[list[float]]:
        vectors = []
        for patch in patches:
            image = _image_candidate(patch)
            text = patch.text or ""
            vectors.append(self._encode([_with_image_token(text) if image else text], images=[image] if image else None)[0])
        return vectors

    def embed_query_patches(self, query: str, max_patches: int = 8) -> list[list[float]]:
        del max_patches
        return [self.embed_text(query)]

    def _encode(self, texts: list[str], images: list[Any] | None = None) -> list[list[float]]:
        self._load_model()
        try:
            import mlx.core as mx

            kwargs: dict[str, Any] = {
                "text": texts,
                "return_tensors": "mlx",
                "padding": True,
                "truncation": True,
                "max_length": self.max_length,
            }
            if images is not None:
                kwargs["images"] = images
            inputs = self._processor(**kwargs)
            inputs = {key: mx.array(value) if not isinstance(value, mx.array) else value for key, value in inputs.items()}
            out = self._model(**inputs)
            embeddings = getattr(out, "text_embeds", out)
            mx.eval(embeddings)
            return [_fit_dimension(row, self.dimension) for row in _as_numpy_rows(embeddings)]
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to encode with {self.model_name}: {exc}") from exc


class MlxColQwenEmbedder(FeatureHashEmbedder):
    """ColQwen MLX token multi-vector provider for late-interaction smoke tests."""

    def __init__(
        self,
        model_name: str = DEFAULT_MLX_COLQWEN_MODEL,
        dimension: int = 128,
        max_patch_vectors: int = 512,
    ):
        super().__init__(dimension=dimension)
        self.backend = "mlx-colqwen"
        self.model_name = model_name
        self.max_patch_vectors = int(max_patch_vectors)
        self._model = None
        self._processor = None
        self._cache_cls = None
        self._image_processor = None
        self._http = httpx.Client(timeout=20, follow_redirects=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_embeddings import load
            from mlx_vlm.models.qwen2_5_vl.language import KVCache
            from transformers import AutoImageProcessor
        except Exception as exc:
            raise MlxEmbedderError(
                "ColQwen MLX dependencies are missing. Run with Python 3.12 and `uv sync --extra mlx`."
            ) from exc
        try:
            self._model, self._processor = load(self.model_name)
            self._cache_cls = KVCache
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to load MLX ColQwen model {self.model_name}: {exc}") from exc

    def embed_text(self, text: str) -> list[float]:
        return mean_pool(self.embed_query_patches(text), self.dimension)

    def embed_record(self, record: OpenRecord) -> list[float]:
        return mean_pool(self.embed_patches(self.patch_record(record)), self.dimension)

    def embed_patches(self, patches: list[Patch]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for patch in patches:
            image = self._patch_image(patch)
            if image is not None:
                vectors.extend(self._encode_image_token_vectors(image, patch.text or ""))
            else:
                vectors.extend(self._encode_token_vectors(patch.text or ""))
            if len(vectors) >= self.max_patch_vectors:
                break
        return vectors[: self.max_patch_vectors] or super().embed_patches(patches)

    def embed_query_patches(self, query: str, max_patches: int = 32) -> list[list[float]]:
        return self._encode_token_vectors(query)[:max_patches]

    def _encode_token_vectors(self, text: str) -> list[list[float]]:
        self._load_model()
        try:
            import mlx.core as mx

            inputs = self._processor(text=[text], return_tensors="mlx", padding=True)
            inputs = {key: mx.array(value) if not isinstance(value, mx.array) else value for key, value in inputs.items()}
            cache = [self._cache_cls() for _ in range(len(self._model.vlm.language_model.model.layers))]
            out = self._model(**inputs, cache=cache)
            embeddings = getattr(out, "text_embeds", out)
            mx.eval(embeddings)
            rows = _as_numpy_rows(embeddings)
            if rows.ndim == 3:
                rows = rows.reshape(-1, rows.shape[-1])
            return [_fit_dimension(row, self.dimension) for row in rows if np.linalg.norm(row) > 1e-8]
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to encode ColQwen vectors with {self.model_name}: {exc}") from exc

    def _encode_image_token_vectors(self, image: Any, text: str) -> list[list[float]]:
        self._load_model()
        try:
            import mlx.core as mx

            image_inputs = self._image_processor(images=[image], return_tensors="np")
            image_grid = image_inputs["image_grid_thw"]
            image_tokens = self._image_token_count(image_grid[0])
            prompt = "<|vision_start|>" + "<|image_pad|>" * image_tokens + f"<|vision_end|> {text}"
            text_inputs = self._processor(text=[prompt], return_tensors="np", padding=True)
            inputs = {**text_inputs, **image_inputs}
            inputs = {key: mx.array(value) if not isinstance(value, mx.array) else value for key, value in inputs.items()}
            cache = [self._cache_cls() for _ in range(len(self._model.vlm.language_model.model.layers))]
            out = self._model(**inputs, cache=cache)
            embeddings = getattr(out, "image_embeds", None)
            if embeddings is None:
                embeddings = getattr(out, "text_embeds", out)
            mx.eval(embeddings)
            rows = _as_numpy_rows(embeddings)
            if rows.ndim == 3:
                rows = rows.reshape(-1, rows.shape[-1])
            return [_fit_dimension(row, self.dimension) for row in rows if np.linalg.norm(row) > 1e-8]
        except Exception as exc:
            raise MlxEmbedderError(f"Failed to encode ColQwen image vectors with {self.model_name}: {exc}") from exc

    def _image_token_count(self, image_grid: Any) -> int:
        t, h, w = [int(value) for value in image_grid]
        merge = int(getattr(self._model.vlm.vision_tower, "spatial_merge_size", 2))
        return max(1, (h // merge) * (w // merge) * t)

    def _patch_image(self, patch: Patch) -> Any | None:
        value = patch.asset_url or patch.source_file
        if not value:
            return None
        try:
            if patch.kind.startswith("pdf"):
                return self._render_pdf_page(value, max(0, (patch.page or 1) - 1))
            if patch.kind.startswith(("visual", "video")):
                return self._load_image(value)
        except Exception:
            return None
        return None

    def _load_image(self, value: str) -> Any:
        from PIL import Image

        if value.startswith(("http://", "https://")):
            import io

            response = self._http.get(value)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        path = Path(value.removeprefix("file://")).expanduser()
        return Image.open(path).convert("RGB")

    def _render_pdf_page(self, value: str, page_index: int) -> Any:
        import tempfile

        import pypdfium2 as pdfium

        pdf_bytes = self._bytes(value)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(pdf_bytes)
            handle.flush()
            pdf = pdfium.PdfDocument(handle.name)
            if len(pdf) == 0:
                raise MlxEmbedderError("PDF has no pages")
            page = pdf[min(page_index, len(pdf) - 1)]
            image = page.render(scale=1.35).to_pil().convert("RGB")
            page.close()
            pdf.close()
        return image

    def _bytes(self, value: str) -> bytes:
        if value.startswith(("http://", "https://")):
            response = self._http.get(value)
            response.raise_for_status()
            return response.content
        path = Path(value.removeprefix("file://")).expanduser()
        return path.read_bytes()


def _with_image_token(text: str) -> str:
    if "<|image_pad|>" in text:
        return text
    return f"<|vision_start|><|image_pad|><|vision_end|> {text}"


def _image_candidate(patch: Patch) -> str:
    if patch.kind.startswith(("visual", "video", "pdf")) and (patch.asset_url or patch.source_file):
        return patch.asset_url or patch.source_file
    return ""


def _as_numpy_rows(value: Any) -> np.ndarray:
    if hasattr(value, "tolist"):
        arr = np.asarray(value.tolist(), dtype=np.float32)
    else:
        arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _fit_dimension(row: np.ndarray, dimension: int | None = None) -> list[float]:
    target = int(dimension or row.shape[-1])
    vec = np.asarray(row, dtype=np.float32)
    if vec.shape[0] > target:
        vec = vec[:target]
    elif vec.shape[0] < target:
        vec = np.pad(vec, (0, target - vec.shape[0]))
    return normalize(vec).astype(float).tolist()


def local_image(path: str) -> Any:
    from PIL import Image

    return Image.open(Path(path).expanduser()).convert("RGB")

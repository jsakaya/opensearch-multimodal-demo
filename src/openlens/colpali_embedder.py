from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from .embeddings import FeatureHashEmbedder, mean_pool
from .models import Patch


MODEL_ALIASES = {
    "colpali": "vidore/colpali-v1.3-hf",
    "colpali-v1.3": "vidore/colpali-v1.3-hf",
    "colpali-v1.2": "vidore/colpali-v1.2-hf",
}


class ColPaliEmbedderError(RuntimeError):
    pass


def colpali_runtime_status() -> dict[str, Any]:
    """Return lightweight dependency and accelerator state without loading the checkpoint."""
    status: dict[str, Any] = {}
    try:
        import torch
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": False,
            "device": "unavailable",
            "colpali_available": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    status.update(
        {
            "torch_available": True,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    )
    try:
        from transformers import ColPaliForRetrieval, ColPaliProcessor  # noqa: F401

        status["colpali_available"] = True
    except Exception as exc:
        status["colpali_available"] = False
        status["detail"] = f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        status.update(
            {
                "device": "cuda",
                "current_device": int(device_id),
                "device_name": torch.cuda.get_device_name(device_id),
                "capability": f"{props.major}.{props.minor}",
                "total_vram_gb": round(total_bytes / 1024**3, 2),
                "free_vram_gb": round(free_bytes / 1024**3, 2),
            }
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        status.update({"device": "mps"})
    else:
        status.update({"device": "cpu"})
    return status


class ColPaliEmbedder(FeatureHashEmbedder):
    """ColPali/ColVision visual-document multi-vector embedder.

    The model emits many 128-dimensional token or patch vectors per page/image.
    OpenLens stores those vectors in OpenSearch as an unindexed source field for
    native `lateInteractionScore` reranking and mean-pools the same vectors into
    the indexed HNSW vector used for first-stage retrieval.
    """

    def __init__(
        self,
        model_name: str = "colpali-v1.3",
        dimension: int = 128,
        batch_size: int = 2,
        max_pages: int = 1,
        max_patch_vectors: int = 1024,
        image_timeout_s: float = 20,
    ):
        super().__init__(dimension=dimension)
        self.backend = "colpali"
        self.model_name = MODEL_ALIASES.get(model_name, model_name)
        self.dimension = int(dimension)
        self.batch_size = max(1, int(batch_size))
        self.max_pages = max(1, int(max_pages))
        self.max_patch_vectors = max(1, int(max_patch_vectors))
        self.image_timeout_s = float(image_timeout_s)
        self._model = None
        self._processor = None
        self._http = httpx.Client(timeout=self.image_timeout_s, follow_redirects=True)
        self._pdf_bytes_cache: dict[str, bytes] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except Exception as exc:
            raise ColPaliEmbedderError(
                "ColPali dependencies are missing. Install with `uv sync --extra colpali`."
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        try:
            self._model = ColPaliForRetrieval.from_pretrained(self.model_name, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self._model = ColPaliForRetrieval.from_pretrained(self.model_name, **load_kwargs)
        except Exception as exc:
            raise ColPaliEmbedderError(f"Failed to load {self.model_name}: {exc}") from exc

        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)

    @property
    def _device(self):
        self._load_model()
        return next(self._model.parameters()).device

    def embed_text(self, text: str) -> list[float]:
        return mean_pool(self.embed_query_patches(text), self.dimension)

    def embed_query_patches(self, query: str, max_patches: int = 8) -> list[list[float]]:
        del max_patches
        return self._encode_texts([query], query=True)[0]

    def embed_patches(self, patches: list[Patch]) -> list[list[float]]:
        objects = [self._patch_to_object(patch) for patch in patches]
        encoded = self._encode_objects(objects)
        flat: list[list[float]] = []
        for vectors in encoded:
            flat.extend(vectors)
        return flat or super().embed_patches(patches)

    def _patch_to_object(self, patch: Patch) -> dict[str, Any]:
        text = patch.text or ""
        url = patch.asset_url or patch.source_file
        if patch.kind.startswith("pdf") and url and (patch.page or 1) <= self.max_pages:
            try:
                return {"image": self._render_pdf_page(url, max(0, (patch.page or 1) - 1)), "text": text}
            except Exception:
                return {"text": text}
        if patch.kind.startswith(("visual", "video")) and url:
            try:
                return {"image": self._load_image(url), "text": text}
            except Exception:
                return {"text": text}
        return {"text": text}

    def _encode_objects(self, objects: list[dict[str, Any]]) -> list[list[list[float]]]:
        results: list[list[list[float]] | None] = [None] * len(objects)
        image_items = [(idx, obj["image"]) for idx, obj in enumerate(objects) if obj.get("image") is not None]
        text_items = [(idx, obj.get("text") or "") for idx, obj in enumerate(objects) if obj.get("image") is None]

        for start in range(0, len(image_items), self.batch_size):
            batch = image_items[start : start + self.batch_size]
            vectors = self._encode_images([image for _idx, image in batch])
            for (idx, _image), value in zip(batch, vectors, strict=False):
                results[idx] = value

        for start in range(0, len(text_items), self.batch_size):
            batch = text_items[start : start + self.batch_size]
            vectors = self._encode_texts([text for _idx, text in batch], query=False)
            for (idx, _text), value in zip(batch, vectors, strict=False):
                results[idx] = value

        return [value or [] for value in results]

    def _encode_images(self, images: list[Any]) -> list[list[list[float]]]:
        self._load_model()
        import torch
        import torch.nn.functional as F

        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        t0 = time.monotonic()
        with torch.no_grad():
            embeddings = F.normalize(self._model(**inputs).embeddings, p=2, dim=-1)
        if os.getenv("OPENLENS_COLPALI_VERBOSE"):
            print(f"colpali encoded {len(images)} images in {time.monotonic() - t0:.2f}s")
        return self._tensor_to_multivectors(embeddings, inputs.get("attention_mask"))

    def _encode_texts(self, texts: list[str], query: bool) -> list[list[list[float]]]:
        self._load_model()
        import torch
        import torch.nn.functional as F

        inputs = self._processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        t0 = time.monotonic()
        with torch.no_grad():
            embeddings = F.normalize(self._model(**inputs).embeddings, p=2, dim=-1)
        if os.getenv("OPENLENS_COLPALI_VERBOSE"):
            role = "queries" if query else "text patches"
            print(f"colpali encoded {len(texts)} {role} in {time.monotonic() - t0:.2f}s")
        return self._tensor_to_multivectors(embeddings, inputs.get("attention_mask"))

    def _tensor_to_multivectors(self, embeddings, attention_mask) -> list[list[list[float]]]:
        values = embeddings.detach().cpu().float()
        masks = attention_mask.detach().cpu() if attention_mask is not None and attention_mask.ndim == 2 else None
        output: list[list[list[float]]] = []
        for row_index, row in enumerate(values):
            if masks is not None and masks.shape[1] == row.shape[0]:
                row = row[masks[row_index].bool()]
            if row.shape[0] > self.max_patch_vectors:
                step = max(1, row.shape[0] // self.max_patch_vectors)
                row = row[::step][: self.max_patch_vectors]
            output.append(row.tolist())
        return output

    def _load_image(self, value: str):
        from PIL import Image

        value = str(value)
        if value.startswith(("http://", "https://")):
            response = self._http.get(value)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        path = Path(value.removeprefix("file://")).expanduser()
        return Image.open(path).convert("RGB")

    def _render_pdf_page(self, value: str, page_index: int):
        from PIL import Image

        try:
            import pypdfium2 as pdfium
        except Exception as exc:
            raise ColPaliEmbedderError("PDF page rendering needs `uv sync --extra colpali`.") from exc

        pdf_bytes = self._pdf_bytes(value)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            handle.write(pdf_bytes)
            handle.flush()
            pdf = pdfium.PdfDocument(handle.name)
            if len(pdf) == 0:
                raise ColPaliEmbedderError("PDF has no pages")
            page = pdf[min(page_index, len(pdf) - 1)]
            pil_image = page.render(scale=1.35).to_pil()
            page.close()
            pdf.close()
        if not isinstance(pil_image, Image.Image):
            raise ColPaliEmbedderError("PDF renderer did not return a PIL image")
        return pil_image.convert("RGB")

    def _pdf_bytes(self, value: str) -> bytes:
        value = str(value)
        if value in self._pdf_bytes_cache:
            return self._pdf_bytes_cache[value]
        if value.startswith(("http://", "https://")):
            response = self._http.get(value)
            response.raise_for_status()
            content = response.content
        else:
            path = Path(value.removeprefix("file://")).expanduser()
            content = path.read_bytes()
        self._pdf_bytes_cache[value] = content
        return content

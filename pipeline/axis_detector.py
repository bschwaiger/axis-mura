#!/usr/bin/env python3
"""
AXIS — Automated X-Ray Identification for the Skeleton.

Single-image inference for AXIS-MURA-v1B (LoRA fine-tune of MedGemma 1.5 4B).
Accepts DICOM (.dcm), PNG, or JPEG. DICOMs are auto-windowed for bone.

Backends:
    mlx          — Apple Silicon via mlx-vlm. 4-bit quantized. Default.
    transformers — HuggingFace Transformers. For CUDA / cloud / non-Apple.

The AXIS-MURA-v1B-4bit weights are not bundled with this repository (see
README for the access flow). Without them, the script runs against the base
MedGemma 1.5 4B for pipeline verification:

    python axis_detector.py --image xray.dcm
    python axis_detector.py --image xray.dcm --backend transformers

Once the AXIS weights are obtained, point `--model` at the local checkpoint:

    python axis_detector.py --image xray.dcm --model /path/to/axis-mura-v1B-4bit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ============================================================
# CONFIG
# ============================================================

# Default base model per backend. AXIS-MURA-v1B weights are not bundled;
# pass `--model /path/to/axis-mura-v1B-4bit` once you have access.
DEFAULT_MODELS = {
    "mlx": "mlx-community/medgemma-1.5-4b-it-4bit",
    "transformers": "google/medgemma-1.5-4b-it",
}


def _detect_default_backend() -> str:
    try:
        import mlx_vlm  # noqa: F401
        return "mlx"
    except ImportError:
        return "transformers"


DEFAULT_BACKEND = _detect_default_backend()


# Single binary-classification prompt used for the manuscript evaluation.
PROMPT = """You are an expert musculoskeletal radiologist. Analyze this X-ray image and determine whether it is normal or abnormal.

Consider all possible musculoskeletal pathologies including fractures, post-surgical hardware, degenerative changes, dislocations, soft tissue abnormalities, and any other findings.

Respond ONLY with valid JSON in this exact format, no other text:
{
    "abnormal": true or false,
    "confidence": 0.0 to 1.0,
    "category": "fracture, hardware, degenerative, dislocation, soft_tissue, other, or null if normal",
    "location": "brief anatomical description, or null if normal",
    "findings": "one-sentence summary of key findings"
}"""


# ============================================================
# MODEL LOADING — TRANSFORMERS BACKEND (singleton)
# ============================================================

_tf_model = None
_tf_processor = None


def _load_transformers(model_id: str):
    global _tf_model, _tf_processor
    if _tf_model is not None:
        return _tf_model, _tf_processor

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print(f"[↓] Loading {model_id} (transformers)…")

    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon GPU)"
        dtype = torch.float32  # float16 produces empty output on MPS for Gemma
    else:
        device_info = "CPU (slow)"
        dtype = torch.float32

    print(f"    Device: {device_info}")

    t0 = time.time()
    _tf_processor = AutoProcessor.from_pretrained(model_id)
    _tf_model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map="auto",
    )
    print(f"[✓] Model loaded in {time.time() - t0:.1f}s\n")
    return _tf_model, _tf_processor


# ============================================================
# MODEL LOADING — MLX BACKEND (singleton)
# ============================================================

_mlx_model = None
_mlx_processor = None


def _load_mlx(model_path: str):
    global _mlx_model, _mlx_processor
    if _mlx_model is not None:
        return _mlx_model, _mlx_processor

    from mlx_vlm import load

    resolved = str(Path(model_path).expanduser())
    if Path(resolved).is_dir():
        display = resolved
    else:
        display = model_path
        resolved = model_path  # let mlx-vlm resolve from HF hub

    print(f"[↓] Loading {display} (mlx)…")
    t0 = time.time()
    _mlx_model, _mlx_processor = load(resolved)
    print(f"[✓] Model loaded in {time.time() - t0:.1f}s\n")
    return _mlx_model, _mlx_processor


# ============================================================
# DICOM HANDLING
# ============================================================

def _apply_dicom_windowing(pixel_array: np.ndarray, ds) -> np.ndarray:
    """Apply DICOM windowing. Handles RescaleSlope/Intercept, PhotometricInterpretation, VOI LUT."""
    img = pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    img = img * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
    else:
        wc = (img.max() + img.min()) / 2
        ww = img.max() - img.min()
        if ww == 0:
            ww = 1

    lower = wc - ww / 2
    upper = wc + ww / 2
    img = np.clip(img, lower, upper)
    img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        img = 255 - img

    return img


def load_image(image_path: str) -> Image.Image:
    """Load DICOM, PNG, or JPEG. Returns PIL.Image (RGB, max 1024 px)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {image_path}")

    is_dicom = path.suffix.lower() in (".dcm", ".dicom")
    if not is_dicom and path.suffix == "":
        try:
            with open(path, "rb") as f:
                f.seek(128)
                is_dicom = f.read(4) == b"DICM"
        except Exception:
            pass

    if is_dicom:
        if not HAS_PYDICOM:
            print("[!] pydicom not installed. Run: pip install pydicom")
            sys.exit(1)
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array
        if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
            pixel_array = pixel_array[0]
        img_array = _apply_dicom_windowing(pixel_array, ds)
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.open(str(path))

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if max(pil_img.size) > 1024:
        pil_img.thumbnail((1024, 1024), Image.LANCZOS)

    return pil_img


# ============================================================
# INFERENCE
# ============================================================

_active_backend: str = DEFAULT_BACKEND
_active_model: str | None = None


def set_backend(backend: str, model: str | None = None):
    global _active_backend, _active_model
    _active_backend = backend
    _active_model = model


def _resolve_model(backend: str, model_override: str | None) -> str:
    if model_override:
        return model_override
    return DEFAULT_MODELS.get(backend, DEFAULT_MODELS["transformers"])


def query_model(image_path: str) -> dict:
    """Run inference on a single image. Returns parsed JSON + metadata."""
    backend = _active_backend
    model_id = _resolve_model(backend, _active_model)
    if backend == "mlx":
        return _query_mlx(image_path, model_id)
    return _query_transformers(image_path, model_id)


def _query_transformers(image_path: str, model_id: str) -> dict:
    import torch

    model, processor = _load_transformers(model_id)
    pil_image = load_image(image_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    elapsed = time.time() - t0

    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    result = _parse_json_response(raw)
    result["_meta"] = {
        "image": str(image_path),
        "model": model_id,
        "backend": "transformers",
        "inference_time_s": round(elapsed, 1),
        "raw_response": raw,
    }
    return result


def _build_suppress_sampler(processor):
    """Suppress Gemma 3 `<unused*>` thinking tokens at decode time.

    After LoRA merge of MedGemma, Gemma 3's extended-thinking mechanism can
    activate, producing `<unused94>thought…` tokens that hijack generation.
    This sampler zeroes out those token logits before argmax sampling.
    """
    import mlx.core as mx

    suppress_ids = []
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for i in range(256):
        tid = tokenizer.convert_tokens_to_ids(f"<unused{i}>")
        if tid != tokenizer.unk_token_id:
            suppress_ids.append(tid)

    if not suppress_ids:
        return None

    _mask_cache = {}

    def suppressing_sampler(logits: mx.array) -> mx.array:
        vocab_size = logits.shape[-1]
        if vocab_size not in _mask_cache:
            keep = [True] * vocab_size
            for tid in suppress_ids:
                if tid < vocab_size:
                    keep[tid] = False
            _mask_cache[vocab_size] = mx.array(keep)
        mask = _mask_cache[vocab_size]
        logits = mx.where(mask, logits, mx.array(-1e9))
        return mx.argmax(logits, axis=-1)

    return suppressing_sampler


_suppress_sampler = None
_suppress_sampler_built = False


def _get_suppress_sampler(processor):
    global _suppress_sampler, _suppress_sampler_built
    if not _suppress_sampler_built:
        _suppress_sampler = _build_suppress_sampler(processor)
        if _suppress_sampler:
            print("[i] suppress_tokens: sampler active for <unused*> tokens")
        _suppress_sampler_built = True
    return _suppress_sampler


def _query_mlx(image_path: str, model_path: str) -> dict:
    import tempfile
    from mlx_vlm import generate

    model, processor = _load_mlx(model_path)
    pil_image = load_image(image_path)

    messages = [{
        "role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }]
    formatted_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pil_image.save(temp_file.name)

    suppress_sampler = _get_suppress_sampler(processor)

    t0 = time.time()
    try:
        gen_kwargs = dict(image=[temp_file.name], max_tokens=512, temp=0.1)
        if suppress_sampler:
            gen_kwargs["sampler"] = suppress_sampler
        gen_result = generate(model, processor, formatted_prompt, **gen_kwargs)
    finally:
        Path(temp_file.name).unlink(missing_ok=True)
    elapsed = time.time() - t0

    raw = gen_result.text
    result = _parse_json_response(raw)
    result["_meta"] = {
        "image": str(image_path),
        "model": model_path,
        "backend": "mlx",
        "inference_time_s": round(elapsed, 1),
        "generation_tps": round(gen_result.generation_tps, 1),
        "prompt_tps": round(gen_result.prompt_tps, 1),
        "peak_memory_gb": round(gen_result.peak_memory, 2),
        "raw_response": raw,
    }
    return result


def _parse_json_response(text: str) -> dict:
    """Extract JSON from model response, handling common formatting issues."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

    if parsed is None:
        return {
            "abnormal": None,
            "confidence": None,
            "findings": f"[PARSE_FAILED] Raw response: {text[:500]}",
        }
    return parsed


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AXIS — single-image inference",
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Path to X-ray (DICOM, PNG, or JPEG)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        choices=["mlx", "transformers"],
                        help=f"Inference backend (default: {DEFAULT_BACKEND})")
    parser.add_argument("--model", "-m", default=None,
                        help="Model path or HF ID (overrides backend default)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show raw model response")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON only (for piping)")
    args = parser.parse_args()

    set_backend(args.backend, args.model)
    model_display = _resolve_model(args.backend, args.model)
    print(f"[i] AXIS | Backend: {args.backend}, Model: {model_display}\n")

    try:
        result = query_model(args.image)
    except Exception as e:
        result = {"error": str(e), "_meta": {"image": args.image}}

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        return

    abnormal = result.get("abnormal")
    call_str = "ABNORMAL" if abnormal else ("NORMAL" if abnormal is not None else "UNCERTAIN")
    confidence = result.get("confidence")
    conf_str = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "N/A"
    meta = result.get("_meta", {})

    print(f"{'=' * 50}")
    print(f"  FILE:        {Path(args.image).name}")
    print(f"  PATHOLOGY:   {call_str}")
    print(f"  CONFIDENCE:  {conf_str}")
    print(f"  FINDINGS:    {result.get('findings', 'N/A')}")
    if result.get("location"):
        print(f"  LOCATION:    {result['location']}")
    if result.get("category"):
        print(f"  CATEGORY:    {result['category']}")
    print(f"  INFERENCE:   {meta.get('inference_time_s', '?')}s")
    print(f"  BACKEND:     {meta.get('backend', '?')}")
    print(f"{'=' * 50}\n")

    if args.verbose:
        print(f"[RAW RESPONSE]\n{meta.get('raw_response', 'N/A')}\n")


if __name__ == "__main__":
    main()

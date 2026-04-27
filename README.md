# AXIS-MURA

Domain-specific LoRA fine-tune of MedGemma 1.5 4B for musculoskeletal radiograph abnormality detection on the public MURA dataset (Stanford ML Group; Rajpurkar et al., 2017). Inference runs 4-bit on Apple Silicon via MLX, or on CUDA via Transformers.

The model was trained, evaluated against base MedGemma 1.5 4B and Gemma 3 4B across MURA's seven anatomies with per-anatomy McNemar tests (Holm–Bonferroni corrected) and bootstrap confidence intervals, and is the inference target wrapped by [axis-agentic](https://github.com/bschwaiger/axis-agentic) — the agent-driven evaluation framework that won the Nvidia track at Agenthon 001.

This repository is the reproducibility surface: the inference and evaluation code that produces the numbers reported in the accompanying paper. See [`MODEL_CARD.md`](MODEL_CARD.md) for architecture, LoRA setup, training data, evaluation, intended use, and limitations.

## What's in here

- `pipeline/axis_detector.py` — single-image inference (MLX 4-bit on Apple Silicon, or Transformers fallback)
- `pipeline/evaluate.py` — evaluation across model conditions, with per-anatomy breakdown, McNemar's tests (Holm–Bonferroni corrected), and bootstrap confidence intervals
- `MODEL_CARD.md` — architecture, training setup, evaluation, intended use, limitations
- `LICENSE` — Apache 2.0 for code

## Quick start

Verify the pipeline end-to-end against base MedGemma 1.5 4B (no AXIS weights needed):

```bash
pip install -r requirements.txt

# Apple Silicon (MLX backend, default)
python pipeline/axis_detector.py --image path/to/xray.dcm

# CUDA / non-Apple
python pipeline/axis_detector.py --image path/to/xray.dcm --backend transformers
```

Run with the AXIS-MURA-v1B-4bit weights:

```bash
python pipeline/axis_detector.py \
    --image path/to/xray.dcm \
    --model /path/to/axis-mura-v1B-4bit
```

## Reproducing the reported numbers

The evaluation script consumes predictions CSVs (one row per radiograph, columns: `image`, `body_part`, `ground_truth`, `prediction`). With CSVs for the three model conditions on a balanced sample:

```bash
python pipeline/evaluate.py \
    --axis-predictions path/to/axis_v1B_predictions.csv \
    --medgemma-predictions path/to/medgemma_predictions.csv \
    --gemma3-predictions path/to/gemma3_predictions.csv \
    --output-dir results/
```

This emits `metrics_overall.json`, `metrics_per_anatomy.json`, `mcnemar.json`, and a human-readable summary. Expected values in [`MODEL_CARD.md`](MODEL_CARD.md).

## Collaborate

Open to research collaboration, evaluation partnerships, and downstream applications — reach out via [LinkedIn](https://www.linkedin.com/in/benedikt-schwaiger/) or [open an issue](https://github.com/bschwaiger/axis-mura/issues/new).

## Citation

Citation block will be added on publication of the accompanying paper.

## License

- **Code:** Apache License 2.0 (see [LICENSE](LICENSE)).
- **Trained weights:** downstream of MedGemma's Health AI Developer Foundations terms of use.
- **MURA dataset:** Stanford AIMI data use agreement.

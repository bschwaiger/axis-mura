# AXIS-MURA

Domain-specific LoRA fine-tune of MedGemma 1.5 4B for musculoskeletal radiograph abnormality detection on the public MURA dataset (Stanford ML Group; Rajpurkar et al., 2017).

This repository accompanies a manuscript currently under peer review at *Radiology: Artificial Intelligence*. Code, methodology, and reported numbers mirror the manuscript. On acceptance, this README will link to the published paper and the model card will be finalized.

## Status

| Component | State |
|---|---|
| Code (training, inference, evaluation) | **Public** (this repo) |
| `MODEL_CARD.md` | **Public**, reflects the manuscript |
| Trained weights (`axis-mura-v1B-4bit`) | **Available upon reasonable request after publication.** Email the corresponding author. |
| Preprint / paper DOI | **TBD** — linked here when available |

This split — open code, gated weights — is standard practice in medical AI pre-publication.

## What this repo contains

- `pipeline/axis_detector.py` — single-image inference (MLX 4-bit on Apple Silicon, or Transformers fallback)
- `pipeline/evaluate.py` — evaluation across model conditions, with per-anatomy breakdown, McNemar's tests (Holm–Bonferroni corrected), and bootstrap confidence intervals
- `MODEL_CARD.md` — architecture, training setup, evaluation, intended use, limitations
- `LICENSE` — Apache 2.0 for code

## What this repo does NOT contain

- Trained AXIS-MURA-v1B weights (gated; see above)
- The training notebook / training code (held back until publication; the manuscript Methods and `MODEL_CARD.md` document the recipe at the methodological level)
- The MURA dataset (download from [Stanford AIMI](https://aimi.stanford.edu/datasets/mura-msk-xrays))

## Quick start (without AXIS weights)

To verify the pipeline runs end-to-end against the **base** MedGemma 1.5 4B:

```bash
pip install -r requirements.txt

# Apple Silicon (MLX backend, default)
python pipeline/axis_detector.py --image path/to/xray.dcm

# CUDA / non-Apple
python pipeline/axis_detector.py --image path/to/xray.dcm --backend transformers
```

Once the AXIS-MURA-v1B-4bit weights are granted:

```bash
python pipeline/axis_detector.py \
    --image path/to/xray.dcm \
    --model /path/to/axis-mura-v1B-4bit
```

## Reproducing the manuscript numbers

The evaluation script consumes predictions CSV outputs from a batch run (one row per radiograph, columns: `image`, `body_part`, `ground_truth`, `prediction`). Once you have CSVs for the three model conditions on a balanced sample, run:

```bash
python pipeline/evaluate.py \
    --axis-predictions path/to/axis_v1B_predictions.csv \
    --medgemma-predictions path/to/medgemma_predictions.csv \
    --gemma3-predictions path/to/gemma3_predictions.csv \
    --output-dir results/
```

This emits `metrics_overall.json`, `metrics_per_anatomy.json`, `mcnemar.json`, and a human-readable summary. See `MODEL_CARD.md` for the expected values.

## Citation

Citation block will be populated after acceptance. Until then, please contact the corresponding author before referencing this work.

## License

- **Code:** Apache License 2.0 (see `LICENSE`).
- **Trained weights** (when released): downstream of MedGemma's Health AI Developer Foundations terms of use; full terms specified at release.
- **MURA dataset:** Stanford AIMI data use agreement (separately governed; see Stanford AIMI).

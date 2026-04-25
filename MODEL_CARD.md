# Model Card — AXIS-MURA-v1B

## Model details

- **Name:** AXIS-MURA-v1B
- **Description:** A LoRA fine-tune of MedGemma 1.5 4B IT for binary classification of musculoskeletal radiographs (normal vs. abnormal), trained on the publicly available MURA training partition.
- **Architecture:** MedGemma 1.5 4B (Gemma 3 4B decoder + MedSigLIP-400M vision encoder). LoRA adapters applied to all attention and feed-forward projections of the language decoder; vision encoder frozen.
- **Trainable parameters:** ~60 M (~1.4 % of total).
- **Inference precision:** 4-bit (MLX backend). Base bf16 LoRA-merged checkpoint produced before quantization.
- **License:** Apache 2.0 (code). Weights TBD on release; downstream of MedGemma HAI-DEF terms of use.

## Training

- **Dataset:** MURA public release (v1.1), training partition only. Patient-disjoint from the validation split used for evaluation.
- **Class balancing:** the minority class was oversampled to a 1:1 ratio (~43,870 image-label pairs).
- **LoRA configuration:** rank = 32, alpha = 64, dropout = 0.05.
- **Optimization:** 3 epochs; effective batch size 16 (per-device 4 × grad-accum 4); learning rate 2 × 10⁻⁵, cosine schedule with 3 % warmup; weight decay 0.01; bf16 precision; seed = 42.
- **Loss:** language-model next-token loss masked to the JSON response only (prompt, system, image tokens excluded from loss).
- **Hardware:** single NVIDIA A100 80 GB (Google Colab Pro).
- **Final checkpoint:** step 8,226 (end of training); final logged training loss 0.036.

## Inference

A single binary-classification prompt is used at inference. The model emits a JSON object: `{"abnormal": bool, "confidence": float, "category": str|null, "location": str|null, "findings": str}`. The full prompt is in `pipeline/axis_detector.py`.

A defensive `suppress_tokens` step removes Gemma 3's `<unused*>` thinking tokens at decode time; without it, post-merge inference can be hijacked by `<unused94>thought…` sequences. This is a known artefact of LoRA-merging Gemma 3 and is documented in the script.

## Intended use

- **Primary:** research evaluation of domain-adapted open-weight medical vision-language models on out-of-distribution radiological tasks.
- **Out of scope:** clinical decision-making, diagnosis, triage, or any production deployment without prospective regulatory clearance and external validation.

## Evaluation

Evaluation on a class- and anatomy-balanced sample of 2,381 MURA validation radiographs.

| Model | MCC (95 % CI) | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Gemma 3 4B (non-medical baseline) | 0.078 [0.039, 0.118] | 94.7 % | 9.4 % | 54.6 % |
| MedGemma 1.5 4B (zero-shot) | 0.433 [0.399, 0.467] | 55.7 % | 85.9 % | 69.9 % |
| **AXIS-MURA-v1B** | **0.655 [0.625, 0.685]** | **77.6 %** | **87.9 %** | **82.4 %** |

Cochran's Q omnibus test across the three models was significant (Q = 439.5, df = 2, P < .001); all pairwise McNemar's tests with Holm–Bonferroni correction were significant (P < .001). Bootstrap MCC confidence intervals are non-overlapping across all three model pairs.

Per-anatomy MCC for AXIS-MURA-v1B ranged from 0.531 (shoulder) to 0.766 (forearm).

## Limitations

- Evaluation was performed on the MURA validation split — same institution and era as the training data. External, multi-site validation is required before any clinical use.
- MURA labels are binary (normal vs. abnormal) and aggregate fractures, surgical hardware, degenerative changes, and other pathologies; per-pathology performance is not characterized.
- The model's emitted "confidence" is a text-generated value, not a calibrated posterior probability; no threshold tuning was performed.
- The vision encoder is frozen and was not pre-trained on musculoskeletal extremity radiographs, identified as the likely performance ceiling.
- 4-bit quantization is used at inference; an exact-precision baseline was not separately characterized.

## Bias and risk

This model inherits biases from MedGemma 1.5 (population, modality, geographic distribution) and from MURA (Stanford institution, 2001–2012 era). Performance on populations or imaging hardware outside that distribution has not been characterized.

## Citation

Placeholder — will be populated on publication.

## Contact

For weight access requests post-publication, see the corresponding author affiliation in the published paper or this repo's README once the citation block is filled in.

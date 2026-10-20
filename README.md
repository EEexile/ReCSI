# ReCSI

**Residual Correction and Speaker-Structured Inference for Audio-Text Emotion Recognition**

ReCSI freezes an audio-text recognizer, learns a residual MLP correction, calibrates its probabilities, and applies exact same-speaker forest inference with TRAIN-estimated class priors and transitions.

## Structure

```text
ReCSI/
├── meld/                  # Stage-I training and model
├── iemocap/               # Stage-I five-fold training and model
├── recsi/                 # Stage-II MLP, calibration, inference and CLI
├── feature_extraction/    # Metadata preparation and Stage-I cache export
├── configs/               # Local-asset configuration templates
├── requirements.txt
└── README.md
```

## Requirements and data

Use Python 3.10+ with PyTorch, NumPy, and the packages in `requirements.txt`. Stage I additionally needs matching torchaudio, transformers and soundfile installations. Run commands from this repository root; no package installation is required.

Prepare MELD/IEMOCAP locally and edit asset paths in `configs/`. CSV columns are `file,text,emotion,speaker`; audio filenames are `<file>.wav`. MELD uses `meld_train_dia0_utt0`-style identifiers and the official train/dev/test splits. IEMOCAP uses original `Ses01...` identifiers and its four-class labels, merging excited into happy. If regenerating CSV files, verify their row/label order and update the expected CSV hashes in the configurations.

Pretrained Wav2Vec2/RoBERTa models and tokenizers must already exist locally. All loads use offline mode and `local_files_only=True`. Models, datasets, checkpoints and feature caches are not bundled.

## Stage I

```bash
python meld/train_meld.py --variant anchor_j --config configs/stage1_meld_seed42.json --output outputs/meld42
python iemocap/train_iemocap.py --variant anchor_j --config configs/stage1_iemocap_seed42.json --fold 1 --output outputs/iem42_fold1
```

Repeat IEMOCAP with folds 2–5. Both datasets train for 60 epochs with seed 42, batch size 16, AdamW learning rate `5e-5`, and emotion CE + `0.1 CTC + 0.1` instance contrastive loss. MELD selects by dev WA; IEMOCAP retains the historical held-out-session WA checkpoint-selection protocol.

## Export frozen features

```bash
python feature_extraction/make_metadata.py --dataset MELD --csv local_assets/metadata/meld_train.csv --split train --output local_assets/metadata/meld_train_routes.json
python feature_extraction/export_stage1_cache.py --dataset MELD --config configs/stage1_meld_seed42.json --checkpoint outputs/meld42/best.pt --metadata local_assets/metadata/meld_train_routes.json --split train --output local_assets/cache/MELD/train.pt
```

Export MELD train/dev/test and IEMOCAP train/heldout caches for each fold. For IEMOCAP metadata, supply `--fold`, `--split train` or `heldout`, and `--timestamps` pointing to a JSON mapping from utterance ID to real `start,end,dialogue,speaker` values. Pass `--fold` to the IEMOCAP cache exporter as well. Use checkpoints you created and trust.

Each cache contains:

```python
{
    "ids": [...],
    "x": ...,                 # FP32 tensor, N × (3072 + C)
    "y": ...,                 # int64 labels
    "label_order": [...],
    "stage1_sha256": "...",   # source checkpoint SHA256
    "metadata": [...]         # id/dialogue/speaker + turn (MELD) or start/end (IEMOCAP)
}
```

Feature order is pooled audio (768), pooled text (768), fused representation (1536), then raw logits (`C=7` on MELD, `C=4` on IEMOCAP). Cache rows and metadata must have identical order.

## Stage II

Set cache paths and evaluation-file SHA256 values in `configs/stage2_primary.json`, then run:

```bash
python -m recsi fit --config configs/stage2_primary.json --output outputs/recsi
python -m recsi evaluate --study outputs/recsi --output outputs/recsi_evaluation --device cpu
```

The primary recipe uses correction initializations 42/43/44. Each initialization averages three shuffle-order member parameter sets. MELD members complete 30 epochs with dev W-F1 selection; IEMOCAP members use one fixed epoch. Temperature fits only MELD-dev NLL and transfers unchanged to IEMOCAP. Normalization, priors and transitions use each training split. Evaluation follows the global selection lock and writes a one-time receipt. Inference uses complete dialogues, including later unlabeled observations.

The inference MLP is `3072+C → 64 ReLU → C`. `compat.py` retains the original training layout. Export an already-averaged historical head to the compact inference layout with:

```bash
python -m recsi compact-head --input outputs/recsi/MELD/head_init42.pt --output outputs/compact_meld42.pt
```

## Results

W-F1 (%), mean ± sample SD across three correction initializations. IEMOCAP first averages five folds within each initialization; Stage-I seed is fixed at 42.

| Dataset | Stage I | ReCSI |
|---|---:|---:|
| MELD | 62.18 | 63.73 ± 0.06 |
| IEMOCAP, five folds | 78.74 | 81.68 ± 0.04 |

For a small synthetic check without datasets or pretrained models:

```bash
python -m recsi demo
```

Stage-I building blocks derive from [MMSD](https://github.com/EEexile/MMSD). See `NOTICE.md` for attribution. This source package was checked with CPU mathematical/synthetic tests; packaging did not rerun the full benchmarks.

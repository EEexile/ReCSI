# ReCSI

**Residual Correction and Speaker-Structured Inference for Audio-Text Emotion Recognition**

ReCSI combines residual logit correction, temperature calibration, and same-speaker emotion inference on top of a frozen audio-text recognizer.

## Results

W-F1 (%), mean ± sample SD across three correction initializations (42/43/44). Stage-I seed is 42; IEMOCAP first averages five folds.

| Dataset | Stage I | ReCSI |
|---|---:|---:|
| MELD | 62.18 | 63.73 ± 0.06 |
| IEMOCAP | 78.74 | 81.68 ± 0.04 |

## Setup

Use Python 3.10+ and install the packages in requirements.txt, with matching PyTorch and torchaudio builds. Prepare MELD/IEMOCAP and local Wav2Vec2/RoBERTa weights and tokenizers; edit paths in configs/. Model loading is offline.

CSV columns are file,text,emotion,speaker; audio filenames are <file>.wav. MELD uses the official train/dev/test splits and meld_train_dia0_utt0-style IDs. IEMOCAP uses Ses01... IDs and four classes, merging excited into happy. Update the configured CSV hashes when regenerating metadata.

## Stage I

~~~bash
python meld/train_meld.py --variant anchor_j --config configs/stage1_meld_seed42.json --output outputs/meld42
python iemocap/train_iemocap.py --variant anchor_j --config configs/stage1_iemocap_seed42.json --fold 1 --output outputs/iem42_fold1
~~~

Run IEMOCAP folds 1–5. Training uses 60 epochs and emotion CE + 0.1 CTC + 0.1 instance contrastive loss. MELD selects by dev WA; IEMOCAP retains held-out-session WA checkpoint selection.

## Frozen features

~~~bash
python feature_extraction/make_metadata.py --dataset MELD --csv local_assets/metadata/meld_train.csv --split train --output local_assets/metadata/meld_train_routes.json
python feature_extraction/export_stage1_cache.py --dataset MELD --config configs/stage1_meld_seed42.json --checkpoint outputs/meld42/best.pt --metadata local_assets/metadata/meld_train_routes.json --split train --output local_assets/cache/MELD/train.pt
~~~

Export MELD train/dev/test and IEMOCAP train/heldout for each fold. For IEMOCAP metadata, pass --fold and --timestamps: a JSON mapping each utterance ID to its true start,end,dialogue,speaker. Also pass --fold to the exporter. Each cache stores ids, x, y, label_order, stage1_sha256, and metadata in matching row order. FP32 features x concatenate audio (768), text (768), fusion (1536), and logits (C=7 or 4). See --help for the metadata and export commands.

## Stage II

Set cache paths and precomputed evaluation-cache SHA256 values in configs/stage2_primary.json.

~~~bash
python -m recsi fit --config configs/stage2_primary.json --output outputs/recsi
python -m recsi evaluate --study outputs/recsi --output outputs/recsi_evaluation --device cpu
~~~

Each correction initialization averages three shuffle-order member parameter sets. MELD members train for 30 epochs with dev W-F1 selection; IEMOCAP members use one fixed epoch. Temperatures fit MELD-dev NLL and transfer unchanged to IEMOCAP. Normalization, priors, and transitions use training data. Inference uses the complete dialogue, including later unlabeled utterances. Evaluation follows the saved selection lock.

The compact inference head is 3072+C → 64 ReLU → C. To convert an existing head after parameter averaging:

~~~bash
python -m recsi compact-head --input outputs/recsi/MELD/head_init42.pt --output outputs/compact_meld42.pt
~~~

The original training layout and its conversion are included in recsi/model.py. recsi/inference.py contains calibration and speaker inference; recsi/data.py handles caches and IO.

Stage I builds on [MMSD](https://github.com/EEexile/MMSD). Temperature scaling follows [Guo et al. (ICML 2017)](https://proceedings.mlr.press/v70/guo17a.html).

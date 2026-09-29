<div align="center">

<h1>MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation  (NeurIPS 2025)</h1>


[Yang Han](https://csyanghan.github.io/)<sup>1,2</sup>, [Pengyu Wang](https://peng-yuwang.github.io/Pengyu-page/)<sup>1,2</sup>, [Kai Yu](https://x-lance.sjtu.edu.cn/~kaiyu/)<sup>1,2</sup>, [Xin Chen](https://openreview.net/profile?id=~xin_chen66)<sup>2</sup>, [Lu Chen](https://coai-sjtu.github.io/)<sup>1, 2</sup>

<sup>1</sup> X-LANCE Lab, Shanghai Jiao Tong University, Shanghai  <sup>2</sup> Suzhou Laboratory, Suzhou.


<a href="https://arxiv.org/abs/2510.20615" target="_blank"  style="margin-right: 4px;"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.20615-b31b1b?logo=arxiv&logoColor=white"/></a>
<a href='https://modelscope.cn/studios/csyanghan/MS-BART/'><img src='https://img.shields.io/badge/ModelScope-Demo-blue'></a>

</div>


> **TL; DR:**  MS-BART is the first to leverage language model for mass spectra structure elucidation by introducing a unified vocabulary and enabling end-to-end pretraining, fine-tuning, and alignment.

<p align="center">
  <img src="images/ms-bart-framework.png">
</p>



## Environment Setup

```bash
uv python install 3.9
uv sync --managed-python
source .venv/bin/activate
```

This creates a Python 3.9 environment in `.venv`. Run commands from the project root.

## Preprocessed Dataset and Model Weights

Download the preprocessed datasets and original model weights from
[Figshare](https://figshare.com/articles/dataset/MS-BART-Model-Weights-Data/30393544)
and place them in `data/`.

The original release uses the following directory layout:

```
data
├─ CANOPUS
│  ├─ mist
│  ├─ model-weights # The final MS-BART model on CANOPUS dataset
|  ├─ pretrain-data # clean pretrain data (filter Tanimoto similarity > 0.5)
|  ├─ pretrained-model # pretrain on clean 4M pretrain dataset
│  ├─ train
│  ├─ test
│  ├─ val
├─ MassSpecGym
│  ├─ mist # retrained with clean CANOPUS dataset
│  ├─ model-weights # The final MS-BART model on MassSpecGym dataset
|  ├─ pretrain-data
|  ├─ pretrained-model
│  ├─ train
│  ├─ test
│  ├─ val
```

### Alternative MIST weights from FRIGID

The fingerprint preprocessing script also supports the `mist_msg.pt` and
`mist_canopus.pt` parameter dictionaries from
[FRIGID](https://github.com/coleygroup/FRIGID). Download and extract its
[pretrained checkpoint archive](https://zenodo.org/records/19685145) with
`aria2c` installed:

```bash
mkdir -p data
aria2c -c -x 16 -s 16 -k 1M \
  --auto-file-renaming=false \
  -d data \
  -o frigid_pretrained_checkpoints.tar.gz \
  'https://zenodo.org/records/19685145/files/frigid_pretrained_checkpoints.tar.gz?download=1'

mkdir data/frigid_pretrained_checkpoints
tar -xzf data/frigid_pretrained_checkpoints.tar.gz \
  -C data/frigid_pretrained_checkpoints \
  --keep-old-files --no-same-owner
```

The two MIST weights are extracted directly into
`data/frigid_pretrained_checkpoints/`. The archive also contains DLM and ICEBERG
weights, which are not needed by `preprocess/fp_pred_main.py`.

| `--weight-type` | Default weight path | Loading configuration |
| --- | --- | --- |
| `checkpoint` (default) | `data/<dataset-name>/mist/mist.ckpt` | Hyperparameters and state dictionary from a full Lightning checkpoint |
| `mist-msg` | `data/frigid_pretrained_checkpoints/mist_msg.pt` | MSG preset: hidden size 640, fragment output size 2048 |
| `mist-canopus` | `data/frigid_pretrained_checkpoints/mist_canopus.pt` | CANOPUS preset: hidden size 512, fragment output size 512 |

Both FRIGID presets predict 4096-bit Morgan fingerprints and use the official
[MSG](https://github.com/coleygroup/FRIGID/blob/main/configs/spec2mol_benchmark_msg.yaml)
and [CANOPUS](https://github.com/coleygroup/FRIGID/blob/main/configs/spec2mol_benchmark_canopus.yaml)
inference settings. Use `--fp-ckpt PATH` to override the selected weight path.
`--weight-type` selects the model; `--dataset-name` independently selects the
input dataset format. Full checkpoint mode should only be used with trusted
files because it loads pickled hyperparameters.

## Preprocessing from Raw Data

### Pretraining molecules

Pretraining data generation reads
`data/MassSpecGym/data/molecules/MassSpecGym_molecules_MCES2_disjoint_with_test_fold_4M.tsv`
and writes generated data to `logs/datasets/MassSpecGym/`. The split files are
`pubchem-4M/train.tsv` and `pubchem-4M/val.tsv` under that directory, as used by
`scripts/pretrain.sh`.

```bash
# Generate pretraining data and reserve 10,000 entries for validation.
uv run --no-sync python preprocess/generate_pretrain_data.py
uv run --no-sync python preprocess/split_pretrain_dataset.py
```

### MassSpecGym spectra and fingerprints

Place the raw table at `data/MassSpecGym/data/MassSpecGym.tsv`. The MGF generator
uses `only_MH = False` by default and writes spectra and MIST labels to
`logs/datasets/MassSpecGym-ALL/`:

```bash
uv run --no-sync python preprocess/generate_mgf_and_lables.py
```

`MassSpecGym-ALL` denotes the input across adduct types. Generate fingerprints
for this full input before applying the training-only adduct filter:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run --no-sync python preprocess/fp_pred_main.py \
  --weight-type mist-msg \
  --dataset-name MassSpecGym \
  --mgf-input logs/datasets/MassSpecGym-ALL/MassSpecGym-ALL.mgf \
  --labels logs/datasets/MassSpecGym-ALL/MassSpecGym-ALL_labels.tsv \
  --source-tsv data/MassSpecGym/data/MassSpecGym.tsv \
  --res-dir logs/datasets/MassSpecGym/mist \
  --output-dir logs/datasets/MassSpecGym \
  --device cuda:0 \
  --batch-size 32 \
  --num-workers 8
```

To use the CANOPUS encoder on the same input, change only `--weight-type` to
`mist-canopus`. To use an original full checkpoint, select
`--weight-type checkpoint --fp-ckpt /path/to/mist.ckpt` instead.

### CANOPUS spectra and fingerprints

The CANOPUS generator expects `labels.tsv`, `splits/canopus_hplus_100_0.tsv`,
and `spec_files/` under `data/canopus_train_export/`.

```bash
uv run --no-sync python preprocess/generate_canopus_and_lables.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run --no-sync python preprocess/fp_pred_main.py \
  --weight-type mist-canopus \
  --dataset-name CANOPUS \
  --mgf-input logs/datasets/CANOPUS/CANOPUS.mgf \
  --labels logs/datasets/CANOPUS/CANOPUS_labels.tsv \
  --source-tsv data/canopus_train_export/labels.tsv \
  --split-tsv data/canopus_train_export/splits/canopus_hplus_100_0.tsv \
  --res-dir logs/datasets/CANOPUS/mist \
  --output-dir logs/datasets/CANOPUS \
  --device cuda:0 \
  --batch-size 32 \
  --num-workers 8
```

### Train, validation, and test splits

Preserve the original MassSpecGym `fold` assignments. Apply the adduct filter
only when producing the training split:

| Output | Fold selection | Adduct selection |
| --- | --- | --- |
| `MassSpecGym/train` | `fold == "train"` | `adduct == "[M+H]+"` only |
| `MassSpecGym/val` | `fold == "val"` | All adducts in the processed input |
| `MassSpecGym/test` | `fold == "test"` | All adducts in the processed input |

Do not filter the full MGF or fingerprint table to `[M+H]+`, as this would also
remove non-protonated spectra from validation and test. The MGF generator still
accepts only its supported adduct list, and fingerprint/SELFIES processing drops
records without predictions or with failed SELFIES conversion; "all adducts"
does not imply that every raw record survives preprocessing. For CANOPUS,
preserve the supplied `split` assignments.

Split the generated MassSpecGym fingerprint table at threshold `0.2`:

```bash
uv run --no-sync python preprocess/prepare_test_data.py \
  --dataset-name MassSpecGym \
  --threshold 0.2
```

These are the defaults when no arguments are supplied. Use `--dataset-name CANOPUS`
for CANOPUS or `--data-dir PATH` to override the input directory, which defaults to
`logs/datasets/<dataset-name>`. Outputs are written to `train/`, `val/`, and `test/`
under that directory. `--threshold` selects an existing fingerprint TSV; it does
not recompute fingerprints.

The intended generated-data layout is shown below; `train/`, `val/`, and `test/`
are produced by the splitting step, not by fingerprint prediction:

```text
logs/datasets/
├── MassSpecGym-ALL/
│   ├── MassSpecGym-ALL.mgf
│   └── MassSpecGym-ALL_labels.tsv
├── MassSpecGym/
│   ├── mist/subforms_fp/
│   ├── MassSpecGym_fps_selfies_threshold_*.tsv
│   ├── pubchem-4M/
│   │   ├── train.tsv
│   │   └── val.tsv
│   ├── train/  # [M+H]+ only
│   ├── val/    # All processed adducts
│   └── test/   # All processed adducts
└── CANOPUS/
    ├── CANOPUS.mgf
    ├── CANOPUS_labels.tsv
    ├── mist/subforms_fp/
    ├── CANOPUS_fps_selfies_threshold_*.tsv
    ├── train/
    ├── val/
    └── test/
```

## Step1: Unified Multi-Task Pretraining on Reliably Computed Fingerprints

The pretraining script loads the unified SELFIES/fingerprint tokenizer from
`./ms-bart-tokenizer`.
The SELFIES token vocabulary comes from
[`moldata/vocab_list/zinc.npy`](https://github.com/zjunlp/MolGen/blob/main/moldata/vocab_list/zinc.npy)
in MolGen (ICLR 2024).

Pretraining saves checkpoints to `./log/pretrained-model`. Finetuning and alignment
save to `./log/<run-name>`, and downstream scripts load checkpoints from these
directories.

```bash
bash scripts/pretrain.sh
```

## Step2: Finetuning on Experimental Spectra

Before launching finetuning or alignment, set each script's model and data paths
to the artifacts you intend to use. The preprocessing examples write to
`logs/datasets/`, while some training and evaluation scripts still reference
the original release under `data/`.

In particular, the MSG finetuning and alignment scripts currently select
`data/MassSpecGym/{train,val}/MassSpecGym_fps_selfies_threshold_0.11.tsv`.
The fingerprint generator does not currently produce threshold `0.11`.
Choose the intended threshold and align preprocessing and training accordingly;
renaming a TSV does not change the threshold used to generate its fingerprints.
The CANOPUS finetuning and alignment scripts use threshold `0.2` under
`logs/datasets/CANOPUS/`.

```bash
bash scripts/msg/finetune.sh

bash scripts/canopus/finetune.sh
```

## Step3: Contrastive Alignment via Chemical Feedback

```bash
bash scripts/msg/align.sh

bash scripts/canopus/align.sh
```

## Evaluation

Check the evaluation paths against the selected split and trained checkpoint
before launching. The current MSG evaluation script has a CANOPUS filename in
its test path and uses `NUM_BEAMS` although it defines `NUM_BEAM`. The CANOPUS
evaluation script reads from `data/CANOPUS/`. These data-path and MSG beam settings
need to be corrected for the generated-data workflow; the entry points are:

```bash
bash scripts/msg/eval.sh

bash scripts/canopus/eval.sh
```

## Contact

If you have any questions, please reach out to csyanghan@sjtu.edu.cn

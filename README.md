# ITGAP: Integration of T-cell Receptor and Gene Expression for Antigen Prediction

**Sarah Li\*, Phi Le\*, Leah Ung, Hai Yang, Andrew Ko, Bridget P. Keenan, Li Zhang**  
University of California, San Francisco | \*Equal contribution

ITGAP is a multimodal framework for TCR–antigen recognition prediction. CDR3 sequences and peptides are encoded with Atchley-factor positional embeddings and compressed via autoencoders. Batch-corrected gene expression is mapped to the TCR latent space through an encoder–decoder (ED) network, yielding a joint GEX–TCR representation for downstream classification.

| Setting | ITGAP (ED) | + V/J genes | GEX-only | TCR-only |
|---|---|---|---|---|
| Unsupervised NMI | **0.51** | — | 0.41 | 0.35 |
| Random split PR-AUC (3:1) | 0.929 | **0.960** | 0.866 | 0.931 |
| TCR-based split PR-AUC (3:1) | **0.821** | 0.845 | 0.750 | 0.717 |
| TCR α/β split PR-AUC (3:1) | **0.895** | 0.911 | 0.782 | 0.767 |
| Clinical cohort (exact-match) | **0.801** PR-AUC | — | — | — |

---

## Repository Contents

This repository covers the supervised prediction pipeline. Unsupervised embedding analyses and clinical application code are maintained separately.

| File | Description |
|---|---|
| `tcr_antigen_prediction_utils.py` | All reusable functions: sequence encoding, autoencoders, ED model, classifiers |
| `tcr_beta_prediction_notebook.ipynb` | CDR3β-only prediction — `random_split` / `tcr_split` |
| `tcr_alpha_beta_prediction_notebook.ipynb` | CDR3α + CDR3β prediction — `random_split` / `tcr_ab_split` |
| `requirements.txt` | Python package versions |

---

## Installation

```bash
git clone https://github.com/mlizhangx/ITGAP.git
cd ITGAP
pip install -r requirements.txt
```

> **Apple Silicon only:** `tensorflow-macos==2.9.0` is in `requirements.txt`. On Linux/Windows, replace it with `tensorflow==2.9.0`.

---

## Usage

### Reproducing paper results

Open either notebook in JupyterLab and set the config variables in Section 1:

```python
NEG_RATIO  = "3_1"        # "3_1" or "5_1"
SPLIT_TYPE = "tcr_split"  # "random_split" | "tcr_split" | "tcr_ab_split" (α/β notebook only)
RUN        = 1            # 1–5
USE_VJ     = False        # True to include V/J gene spectral embeddings
```

Run all cells. Each notebook trains three models in sequence: Batch GEX + Peptide, TCR + Peptide, and ED (GEX → TCR) + Peptide. Sequence embeddings and ED latents are cached to disk on first run and reloaded automatically thereafter.

### Generating splits for a new dataset

To generate split index files for your own data:

```python
from tcr_antigen_prediction_utils import create_random_splits

create_random_splits(ref_data, labels, output_dir="./output/", n_runs=5)
```

For TCR-grouped splits (no CDR3β overlap between train and test):

```python
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

gss = GroupShuffleSplit(n_splits=5, test_size=0.30, random_state=42)
for run, (train_idx, test_idx) in enumerate(gss.split(ref_data, groups=ref_data["tcr"]), start=1):
    np.save(f"output/tcr_split/run{run}_train.npy", train_idx)
    np.save(f"output/tcr_split/run{run}_test.npy",  test_idx)
```

### Using the utility functions directly

```python
from tcr_antigen_prediction_utils import (
    load_dataset, load_split,
    load_atchley, PositionalEmbedding, build_fit_pe_ae,
    build_ed_integration_model, fit_ed_per_split,
    assemble_features, build_residual_classifier, compile_and_train,
    evaluate_classifier,
)

data  = load_dataset(neg_ratio="3_1")
split = load_split(data["output_dir"], "tcr_split", run=1,
                   ref_data=data["ref_data"], ref_data_alpha_beta=data["ref_data"],
                   labels=data["labels"])

word_vecs, aa_conv = load_atchley()
pt = PositionalEmbedding(d_model=6, word_vectors=word_vecs, length=35)
emb_train, emb_val, emb_test = build_fit_pe_ae(
    split["train_raw"]["tcr"].values,
    split["val_raw"]["tcr"].values,
    split["test_raw"]["tcr"].values,
    pt=pt, index_aa_converter=aa_conv,
    latent_dim=128, length=35, d_model=6, epochs=50,
    ae_name="tcr_beta_ae",
    save_paths={"train": "out/tcr_train.npy", "val": "out/tcr_val.npy", "test": "out/tcr_test.npy"},
)

X_train, X_val, X_test = assemble_features(
    (emb_train, emb_val, emb_test),
    (pep_train, pep_val, pep_test),
)
model = build_residual_classifier(X_train.shape[1], name="itgap")
compile_and_train(model, X_train, split["train_labels"], X_val, split["val_labels"])
metrics = evaluate_classifier(model, X_test, split["test_labels"])
print(metrics)  # {"roc_auc": ..., "pr_auc": ...}
```

### Input data format (own dataset)

| File | Format | Required columns |
|---|---|---|
| Reference pairs | CSV | `tcr` (CDR3β), `peptide`, `label` (0/1), `tcr_source_index` |
| Batch-corrected GEX | CSV | cells × PCs, row-indexed by `tcr_source_index` |
| AnnData | `.h5ad` | used for donor metadata |
| V/J embeddings *(optional)* | CSV | row-indexed by `tcr_source_index` |

---

## Data

- **10x Genomics benchmark** — 145,479 CD8+ T cells, 4 donors, 9 pMHC-validated peptides: publicly available from 10x Genomics.
- **Neoadjuvant CD40 agonism cohort** — 16,286 T cells, E/GEJ cancer patients: [GSE244748](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244748).

---

## Citation

Manuscript submitted to *Nature Communications*. Citation will be updated upon publication.

---

## Contact

Open an issue or email [li.zhang@ucsf.edu](mailto:li.zhang@ucsf.edu).

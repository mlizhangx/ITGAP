# ITGAP: Integration of TCR and Gene Expression for Antigen Prediction

**Sarah Li\*, Phi Le\*, Leah Ung, Hai Yang, Bridget P. Keenan, Li Zhang**  
University of California, San Francisco  
\*Equal contribution

ITGAP is a multimodal framework for predicting TCR–antigen recognition by jointly modeling CDR3 sequences and single-cell gene expression. It uses Atchley-factor positional encoding to represent amino acid sequences, compresses them with a sequence autoencoder, and links batch-corrected gene expression to those latent features via an encoder–decoder (ED) network. The resulting integrated embeddings are used for downstream TCR–peptide recognition prediction.

---

## Overview

T cells recognize antigens through highly diverse T-cell receptors (TCRs). Advances in single-cell technologies now enable simultaneous profiling of paired TCR sequences and transcriptomes, but methods that effectively integrate these modalities for antigen prediction remain limited. ITGAP addresses this by learning a joint GEX–TCR latent space that captures both functional and sequence-derived aspects of T-cell responses.

**Key results (10x Genomics benchmark):**
- Unsupervised: NMI of 0.51 vs. 0.41 (GEX-only) and 0.35 (TCR-only)
- Prediction (random split, 3:1 ratio): PR-AUC = 0.934 (0.961 with V/J genes)
- Prediction (TCR-based split, 3:1 ratio): PR-AUC = 0.821 (0.845 with V/J genes) vs. 0.750 GEX-only, 0.717 TCR-only
- Real-world clinical application (neoadjuvant CD40 agonism cohort): PR-AUC = 0.80 on exact-match TCR–antigen pairs from IEDB / McPAS / VDJdb

---

## This Repository

This repository contains the **supervised prediction pipeline** — the modules for encoding TCR and peptide sequences, integrating GEX via the encoder–decoder model, and training/evaluating recognition classifiers. Unsupervised embedding analyses and real-world clinical application code will be released in a companion repository.

### Files

| File | Description |
|---|---|
| `tcr_antigen_prediction_utils.py` | All reusable functions and classes (sequence encoding, AE, ED, classifiers) |
| `tcr_beta_prediction_notebook.ipynb` | End-to-end prediction using CDR3β only (`random_split` / `tcr_split`) |
| `tcr_alpha_beta_prediction_notebook.ipynb` | End-to-end prediction using CDR3α + CDR3β (`random_split` / `tcr_ab_split`) |
| `requirements.txt` | Python package versions from the `tcr_env` conda environment |

---

## Methods Summary

### 1. Sequence Encoding (Atchley + Positional Encoding + Autoencoder)

TCR CDR3 and peptide sequences are encoded using Atchley factors — a set of five physicochemical properties per amino acid. A modified sinusoidal positional encoding (adapted from NLP transformer architectures) is applied to preserve position-specific amino acid context. All sequences are padded/truncated to a fixed length (35 residues for TCR, 15 for peptide).

The positional embeddings are then compressed by a feed-forward autoencoder trained with MSE reconstruction loss, yielding:
- **128-dim** latent embeddings for TCR CDR3β (and CDR3α)
- **64-dim** latent embeddings for peptide antigen

To prevent information leakage, autoencoders are trained separately on train, validation, and test partitions.

### 2. Encoder–Decoder (ED) Integration

A feedforward ED network maps batch-corrected GEX (50-dim PCA + Harmony embeddings) to the TCR autoencoder latent space. The encoder produces a **128-dim** joint GEX–TCR representation; the decoder reconstructs the TCR latent embedding from gene expression, trained with MAE loss. This latent representation forms the multimodal integration embedding used for downstream prediction.

Optionally, V and J gene spectral embeddings (11-dim each) can be concatenated to the final feature vector (`USE_VJ = True`).

### 3. Recognition Prediction

A residual MLP classifier takes concatenated multimodal embeddings (ED latent + peptide AE) as input and outputs a probability of TCR–peptide recognition. The architecture uses residual blocks with batch normalization, dropout (p=0.3), and L1/L2 regularization. Training uses binary cross-entropy with Adam (lr=1e-4) and early stopping on PR-AUC.

### 4. Data Splitting

Two split strategies are supported for rigorous evaluation:
- **`random_split`** — standard stratified random split; measures interpolation performance
- **`tcr_split`** — TCR sequences assigned exclusively to train or test; no CDR3β overlap across partitions; measures generalization to unseen receptors
- **`tcr_ab_split`** — same as above but for paired CDR3α + CDR3β (alpha-beta notebook)

Runs 1–5 are provided with pre-saved index arrays for full reproducibility.

---

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

> **Note:** `tensorflow-macos==2.9.0` is Apple Silicon-specific. Linux/Windows users should substitute `tensorflow==2.9.0`.

### Running on the paper's data

Set configuration at the top of either notebook:

```python
NEG_RATIO  = "3_1"        # "3_1" or "5_1" negative-to-positive ratio
SPLIT_TYPE = "tcr_split"  # "random_split" | "tcr_split" (beta) or "tcr_ab_split" (alpha-beta)
RUN        = 1            # pre-saved split index: 1–5
USE_VJ     = False        # include V/J gene spectral embeddings in the ED model
```

Then run all cells. Sequence embeddings and ED latents are cached to disk after the first run and reloaded automatically on subsequent runs.

### Running on your own dataset

You will need:
1. A reference CSV with columns: `tcr` (CDR3β), `peptide`, `label` (0/1), `tcr_source_index`
2. A batch-corrected GEX matrix (cells × PCs), rows indexed by `tcr_source_index`
3. An AnnData `.h5ad` file (used to extract donor metadata)
4. *(Optional)* V/J gene spectral embeddings CSV indexed by `tcr_source_index`

Generate random split index files once before running:

```python
from tcr_antigen_prediction_utils import create_random_splits
create_random_splits(ref_data, labels, output_dir="./output/", n_runs=5)
```

For TCR-grouped splits on your own data, group rows by CDR3β sequence and apply `sklearn.model_selection.GroupShuffleSplit`, then save the indices with `np.save` following the naming convention in `load_split()`.

---

## Data Availability

The 10x Genomics single-cell dataset (145,479 CD8+ T cells, 4 donors, 9 pMHC multimer-validated peptides) is publicly available from 10x Genomics. The neoadjuvant CD40 agonism clinical dataset is described in the manuscript.

---

## Citation

If you use ITGAP in your research, please cite our manuscript (link to be added upon publication).

**Key dependencies:**
- Atchley WR et al. (2005). Solving the protein sequence metric problem. *PNAS* 102(18):6395–6400.
- Korsunsky I et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature Methods* 16:1289–1296.

---

## Contact

For questions, please open an issue or contact Li Zhang at [li.zhang@ucsf.edu](mailto:li.zhang@ucsf.edu).

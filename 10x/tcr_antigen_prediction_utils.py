import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Activation, Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import layers


# =============================================================================
# Data loading
# =============================================================================

def load_dataset(
    neg_ratio: str = "3_1",
    h5ad_path: str = "./data/merge_gex_all_donors_all_peptides_meta.h5ad",
    batch_gex_path: str = "./data/batch_gex/10X_data_pca_harmony_batch_correction_by_donor_embeddings.csv",
    vj_path: str = "./data/vj_genes/vj_beta_spectral_embeddings.csv",
    output_dir: str = None,
) -> dict:
    """Load all data sources needed for TCR-antigen prediction.

    Returns a dict with keys:
        ref_data, ref_merged, gex, batch_gex,
        ref_data_alpha_beta, vj_genes, labels, output_dir

    Note: h5ad_path is only required to extract raw GEX and alpha-chain sequences.
    It is not included in the repository due to file size; download from 10x Genomics
    (see README). If the file is absent, gex and alpha-chain columns will be unavailable.
    """
    if output_dir is None:
        output_dir = Path(f"./data/neg_ratio_{neg_ratio}")
    else:
        output_dir = Path(output_dir)

    ref_data = pd.read_csv(f"{output_dir}/combined_tcr_peptide_dataset.csv")

    adata = sc.read_h5ad(h5ad_path)

    # Raw GEX matrix (kept for completeness; not used in current model sections)
    ref_gex = adata[ref_data["tcr_source_index"].values, :]
    gex = ref_gex.X.toarray()

    # Build obs lookup and merge donor info
    obs = adata.obs.copy()
    obs = obs.reset_index().reset_index().rename(columns={"index": "tcr_source_index"})
    obs_lookup = obs[["tcr_source_index", "donor"]]

    ref_merged = ref_data.merge(obs_lookup, on="tcr_source_index", how="left",
                                suffixes=("", "_from_obs"))
    ref_merged["donor"] = ref_merged["donor"].fillna(ref_merged["donor_from_obs"])
    ref_merged = ref_merged.drop(columns=["donor_from_obs"])

    # Batch-corrected GEX (PCA + Harmony)
    batch_gex = pd.read_csv(batch_gex_path, index_col=0)
    batch_gex = batch_gex.loc[ref_merged["tcr_source_index"].values]
    batch_gex = batch_gex.reset_index(drop=True)

    # Alpha chain lookup → ref_data_alpha_beta
    tcr_alpha_lookup = (
        obs[["tcr_source_index", "cdr3_TRA"]]
        .dropna(subset=["cdr3_TRA"])
        .drop_duplicates(subset=["tcr_source_index"])
        .rename(columns={"cdr3_TRA": "tcr_alpha"})
    )
    ref_data_alpha_beta = ref_data.merge(
        tcr_alpha_lookup, on="tcr_source_index", how="left"
    )

    # V/J gene spectral embeddings
    vj_gene = pd.read_csv(vj_path)
    vj_genes = vj_gene.loc[ref_data["tcr_source_index"].values]

    labels = ref_data["label"].values

    print(f"Loaded {len(ref_data):,} samples  |  "
          f"pos={ref_data['label'].mean():.1%}  |  output_dir={output_dir}")

    return dict(
        ref_data=ref_data,
        ref_merged=ref_merged,
        gex=gex,
        batch_gex=batch_gex,
        ref_data_alpha_beta=ref_data_alpha_beta,
        vj_genes=vj_genes,
        labels=labels,
        output_dir=output_dir,
    )


# =============================================================================
# Split utilities
# =============================================================================

def _make_split_dict(train_indices, val_indices, test_indices,
                     ref_data, ref_data_alpha_beta, labels):
    return dict(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_raw=ref_data.iloc[train_indices],
        val_raw=ref_data.iloc[val_indices],
        test_raw=ref_data.iloc[test_indices],
        train_raw_ab=ref_data_alpha_beta.iloc[train_indices],
        val_raw_ab=ref_data_alpha_beta.iloc[val_indices],
        test_raw_ab=ref_data_alpha_beta.iloc[test_indices],
        train_labels=labels[train_indices],
        val_labels=labels[val_indices],
        test_labels=labels[test_indices],
    )


def load_split(
    output_dir,
    split_type: str,
    run: int,
    ref_data,
    ref_data_alpha_beta,
    labels,
) -> dict:
    """Load pre-saved train/val/test index arrays.

    split_type: "random_split" | "tcr_split" | "tcr_ab_split"
    run: integer 1-5
    """
    idx_dir = Path(output_dir) / "split_indices"
    train_indices = np.load(idx_dir / f"train_indices_{split_type}_{run}.npy")
    val_indices   = np.load(idx_dir / f"val_indices_{split_type}_{run}.npy")
    test_indices  = np.load(idx_dir / f"test_indices_{split_type}_{run}.npy")

    print(f"split={split_type}  run={run}  |  "
          f"train={len(train_indices):,}  val={len(val_indices):,}  test={len(test_indices):,}")

    return _make_split_dict(
        train_indices, val_indices, test_indices,
        ref_data, ref_data_alpha_beta, labels,
    )


def create_random_splits(ref_data, labels, output_dir, n_runs: int = 5,
                         val_size: float = 0.15, test_size: float = 0.15,
                         random_state: int = 42) -> None:
    """Generate n_runs stratified random train/val/test splits and save index files.

    Useful for running the pipeline on a new dataset. Splits are saved to
    {output_dir}/split_indices/ using the standard naming convention so that
    load_split() can read them directly with split_type="random_split".
    """
    output_dir = Path(output_dir)
    idx_dir = output_dir / "split_indices"
    idx_dir.mkdir(parents=True, exist_ok=True)

    all_idx = np.arange(len(ref_data))
    for run in range(1, n_runs + 1):
        seed = random_state + run
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=val_size + test_size,
            stratify=labels, random_state=seed,
        )
        relative_test = test_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=relative_test,
            stratify=labels[temp_idx], random_state=seed,
        )
        np.save(idx_dir / f"train_indices_random_split_{run}.npy", train_idx)
        np.save(idx_dir / f"val_indices_random_split_{run}.npy",   val_idx)
        np.save(idx_dir / f"test_indices_random_split_{run}.npy",  test_idx)
        print(f"run={run}  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    print(f"Saved {n_runs} random splits → {output_dir}/split_indices/")


# =============================================================================
# Atchley / Positional Encoding
# =============================================================================

def load_atchley(atchley_path: str = "./data/atchley.txt"):
    """Load Atchley factors.

    Returns (word_vectors, index_aa_converter) where
      word_vectors: (21, 6) float32 array (row 0 = padding token)
      index_aa_converter: DataFrame mapping aa letter → integer index (0-based)
    """
    atchley = pd.read_csv(atchley_path, sep="\t")

    index_aa_converter = pd.DataFrame(
        {"aa": atchley["amino.acid"].values, "pos": list(range(20))}
    )
    index_aa_converter.index = index_aa_converter["aa"].values
    index_aa_converter.pop("aa")

    atchley.pop("amino.acid")
    for col in atchley.columns:
        atchley[col] = pd.to_numeric(
            atchley[col].str.replace("−", "-"), errors="coerce"
        )
    atchley["avg"] = atchley.sum(axis=1)

    atchley_aa = pd.concat([index_aa_converter.reset_index(), atchley], axis=1)
    # prepend padding row of zeros
    atchley_amino_prop = pd.concat(
        [pd.DataFrame([[0] * atchley_aa.shape[1]], columns=atchley_aa.columns), atchley_aa],
        axis=0,
    )
    atchley_amino_prop.iloc[0, 0] = "*"

    word_vectors = np.array(atchley_amino_prop.iloc[:, 2:8])  # 6 Atchley factors
    return word_vectors, index_aa_converter


def get_atchley(receptor_seq: str, index_aa_converter, length: int = 35) -> np.ndarray:
    """Convert an amino acid string to a padded integer index array of shape (length,)."""
    res = np.zeros(length, dtype="int")
    for l in range(min(len(receptor_seq), length)):
        aa = receptor_seq[l]
        if aa in index_aa_converter.index:
            res[l] = 1 + int(index_aa_converter.loc[aa].values[0])
        else:
            res[l] = np.random.randint(0, 20)
    return res


def positional_encoding(depth: int, length: int = 25) -> tf.Tensor:
    """Compute sinusoidal positional encoding matrix of shape (length, depth)."""
    positions = np.arange(length)[:, np.newaxis]
    positions = np.repeat(positions, depth, axis=1)
    angle_rates = 1 / 1000
    angle_rads = positions * angle_rates
    s = np.sin(angle_rads)[::2]
    c = 1 - np.cos(angle_rads)[1::2]
    pos_encoding = np.vstack([s, c])
    return tf.cast(pos_encoding, dtype=tf.float32)


class AAEmbedding(tf.keras.layers.Layer):
    """Lookup Atchley embedding for each amino acid index."""

    def __init__(self, word_vectors, **kwargs):
        super().__init__(**kwargs)
        self.word_vectors = tf.constant(word_vectors, dtype=tf.float32)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.word_vectors, inputs)


class PositionalEmbedding(tf.keras.layers.Layer):
    """Combine Atchley AA embedding with positional encoding."""

    def __init__(self, d_model: int, word_vectors, length: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.length = length
        self.embedding = AAEmbedding(word_vectors)
        self.pos_encoding = positional_encoding(depth=d_model, length=length)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, : self.length, :]
        return x


def encode_sequences(sequences, pt: PositionalEmbedding, index_aa_converter,
                     length: int, d_model: int) -> np.ndarray:
    """Encode a list of AA strings → (N, length * d_model) float32 array."""
    X = np.zeros((len(sequences), length * d_model), dtype=np.float32)
    for i, s in enumerate(sequences):
        enc = get_atchley(s, index_aa_converter, length=length)
        X[i] = pt(enc).numpy().reshape(-1)
    return X


# =============================================================================
# Autoencoder utilities
# =============================================================================

def build_sequence_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_dim: int = 164,
    name: str = "seq_ae",
) -> keras.Model:
    """Simple encoder-decoder autoencoder with one hidden layer and BatchNorm."""
    inp = keras.Input(shape=(input_dim,), name="encoder_input")
    x = Dense(hidden_dim)(inp)
    x = BatchNormalization()(x)
    latent = Dense(latent_dim, activation="relu", name="latent")(x)

    x = Dense(hidden_dim)(latent)
    x = BatchNormalization()(x)
    out = Dense(input_dim, activation="linear", name="decoder_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_autoencoder(model, X: np.ndarray, epochs: int = 50, batch_size: int = 64):
    """Fit the autoencoder on X (reconstruction task). Returns Keras history."""
    return model.fit(X, X, epochs=epochs, batch_size=batch_size,
                     validation_data=(X, X), verbose=0)


def get_latent_embeddings(
    model: keras.Model,
    X: np.ndarray,
    latent_layer: str = "latent",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract latent representations from a trained autoencoder."""
    latent_model = keras.Model(model.input, model.get_layer(latent_layer).output)
    return latent_model.predict(X, batch_size=batch_size, verbose=0)


def get_embedding_paths(output_dir, split_type: str, run: int, chain_name: str) -> dict:
    """Return standard file paths for PE+AE embeddings.

    chain_name examples: "TCR" (beta), "TCR_alpha", "peptide"
    Keys returned: "train", "val", "test"
    """
    base = Path(output_dir) / "merged_embeddings"
    tag = f"{split_type}_{run}"
    return {
        "train": str(base / f"train_{chain_name}_with_PE_and_AE_embeddings_{tag}.npy"),
        "val":   str(base / f"val_{chain_name}_with_PE_and_AE_embeddings_{tag}.npy"),
        "test":  str(base / f"test_{chain_name}_with_PE_and_AE_embeddings_{tag}.npy"),
    }


def get_ed_embedding_paths(output_dir, split_type: str, run: int,
                           chain_name: str = "tcr") -> dict:
    """Return standard file paths for ED integration latent embeddings.

    chain_name: "beta" or "alpha"
    Keys returned: "train", "val", "test"
    """
    base = Path(output_dir) / "merged_embeddings"
    tag = f"{split_type}_{run}"
    suffix = f"_alpha" if chain_name == "alpha" else ""
    return {
        "train": str(base / f"ED_integration_batch_gex_tcr{suffix}_embeddings_{tag}.npy"),
        "val":   str(base / f"ED_integration_batch_gex_tcr{suffix}_embeddings_val_{tag}.npy"),
        "test":  str(base / f"ED_integration_batch_gex_tcr{suffix}_embeddings_test_{tag}.npy"),
    }


def build_fit_pe_ae(
    sequences_train,
    sequences_val,
    sequences_test,
    pt: PositionalEmbedding,
    index_aa_converter,
    latent_dim: int,
    length: int,
    d_model: int = 6,
    epochs: int = 50,
    batch_size: int = 64,
    ae_name: str = "seq_ae",
    save_paths: dict = None,
):
    """Encode sequences with PE then compress with an autoencoder.

    Fits a separate AE on each split (train, val, test).
    Returns (emb_train, emb_val, emb_test) each of shape (N, latent_dim).

    If save_paths is provided (dict with "train", "val", "test" → file paths) and all
    files already exist, embeddings are loaded from disk instead of recomputed.
    Otherwise they are computed and, if save_paths is given, saved to disk.
    """
    if save_paths and all(os.path.exists(p) for p in save_paths.values()):
        print(f"  Loading cached embeddings from {Path(save_paths['train']).parent.name}/...")
        return (np.load(save_paths["train"]),
                np.load(save_paths["val"]),
                np.load(save_paths["test"]))

    X_train = encode_sequences(sequences_train, pt, index_aa_converter, length, d_model)
    X_val   = encode_sequences(sequences_val,   pt, index_aa_converter, length, d_model)
    X_test  = encode_sequences(sequences_test,  pt, index_aa_converter, length, d_model)

    input_dim = X_train.shape[1]
    ae = build_sequence_autoencoder(input_dim, latent_dim, name=ae_name)

    fit_autoencoder(ae, X_train, epochs=epochs, batch_size=batch_size)
    emb_train = get_latent_embeddings(ae, X_train)

    fit_autoencoder(ae, X_val, epochs=epochs, batch_size=batch_size)
    emb_val = get_latent_embeddings(ae, X_val, batch_size=64)

    fit_autoencoder(ae, X_test, epochs=epochs, batch_size=batch_size)
    emb_test = get_latent_embeddings(ae, X_test, batch_size=64)

    if save_paths:
        np.save(save_paths["train"], emb_train)
        np.save(save_paths["val"],   emb_val)
        np.save(save_paths["test"],  emb_test)
        print(f"  Saved embeddings → {Path(save_paths['train']).parent.name}/")

    return emb_train, emb_val, emb_test


# =============================================================================
# Encoder-Decoder (ED) integration model
# =============================================================================

def build_ed_integration_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
) -> keras.Model:
    """GEX → TCR-PE encoder-decoder integration model.

    Encoder output (latent_layer) is used as the integrated embedding.
    """
    inp = layers.Input(shape=(input_dim,), name="batch_gex_input")

    x = layers.Dense(hidden_dim, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(hidden_dim // 2, activation="tanh")(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(hidden_dim // 2, name="latent_layer")(x)

    d = layers.Dense(hidden_dim // 2, activation="relu")(latent)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(hidden_dim, activation="tanh")(d)
    d = layers.BatchNormalization()(d)
    out = layers.Dense(output_dim, name="pe_tcr_pred")(d)

    model = Model(inp, out, name="batch_gene2pe_tcr")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae"])
    return model


def fit_ed_per_split(
    model,
    X_train, Y_train,
    X_val,   Y_val,
    X_test,  Y_test,
    epochs: int = 75,
    batch_size: int = 256,
    save_paths: dict = None,
):
    """Fit the ED model independently on train, val, test and extract latent embeddings.

    Returns (emb_train, emb_val, emb_test).

    If save_paths is provided (dict with "train", "val", "test" → file paths) and all
    files already exist, embeddings are loaded from disk without re-training.
    Otherwise they are computed and, if save_paths is given, saved to disk.
    """
    if save_paths and all(os.path.exists(p) for p in save_paths.values()):
        print(f"  Loading cached ED embeddings from {Path(save_paths['train']).name}...")
        return (np.load(save_paths["train"]),
                np.load(save_paths["val"]),
                np.load(save_paths["test"]))

    latent_extractor = tf.keras.Model(
        model.input, model.get_layer("latent_layer").output
    )

    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)
    emb_train = latent_extractor.predict(X_train, batch_size=32, verbose=0)

    model.fit(X_val, Y_val, validation_data=(X_val, Y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)
    emb_val = latent_extractor.predict(X_val, batch_size=128, verbose=0)

    model.fit(X_test, Y_test, validation_data=(X_test, Y_test),
              epochs=epochs, batch_size=batch_size, verbose=0)
    emb_test = latent_extractor.predict(X_test, batch_size=32, verbose=0)

    if save_paths:
        np.save(save_paths["train"], emb_train)
        np.save(save_paths["val"],   emb_val)
        np.save(save_paths["test"],  emb_test)

    return emb_train, emb_val, emb_test


# =============================================================================
# Feature assembly
# =============================================================================

def assemble_features(*split_arrays, normalize: bool = True):
    """Horizontally stack feature arrays for train/val/test and optionally normalize.

    Each argument must be a (train_arr, val_arr, test_arr) tuple (array-like).
    Normalization is min-max per split independently (epsilon = 1e-8).
    Returns (X_train, X_val, X_test) as float32 numpy arrays.
    """
    def _to_array(x):
        if isinstance(x, pd.DataFrame):
            return x.values.astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    trains = [_to_array(t) for t, _, _ in split_arrays]
    vals   = [_to_array(v) for _, v, _ in split_arrays]
    tests  = [_to_array(te) for _, _, te in split_arrays]

    X_train = np.hstack(trains)
    X_val   = np.hstack(vals)
    X_test  = np.hstack(tests)

    if normalize:
        mins_tr = X_train.min(axis=0); maxs_tr = X_train.max(axis=0)
        mins_v  = X_val.min(axis=0);   maxs_v  = X_val.max(axis=0)
        mins_te = X_test.min(axis=0);  maxs_te = X_test.max(axis=0)

        X_train = (X_train - mins_tr) / (maxs_tr - mins_tr + 1e-8)
        X_val   = (X_val   - mins_v)  / (maxs_v  - mins_v  + 1e-8)
        X_test  = (X_test  - mins_te) / (maxs_te - mins_te + 1e-8)

    return X_train, X_val, X_test


# =============================================================================
# Residual MLP classifier
# =============================================================================

def build_residual_mlp(
    input_dim: int,
    dropout_rate: float = 0.3,
    l1_reg: float = 1e-5,
    l2_reg: float = 1e-4,
    name: str = "residual_mlp",
) -> keras.Model:
    """Residual multilayer perceptron: 512→512→256→256→128→64→sigmoid.

    Used for all integration models (ED, TCR-AE, mvTCR, Tessa).
    """
    reg = l1_l2(l1=l1_reg, l2=l2_reg)
    inp = Input(shape=(input_dim,))

    x = Dense(512, kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(dropout_rate)(x)

    res1 = x
    x = Dense(512, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Add()([x, res1]); x = Dropout(dropout_rate)(x)

    x = Dense(256, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(dropout_rate)(x)

    res2 = x
    x = Dense(256, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Add()([x, res2]); x = Dropout(dropout_rate)(x)

    x = Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Dropout(dropout_rate)(x)

    x = Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Dropout(dropout_rate / 2)(x)

    out = Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name=name)


def build_small_residual_mlp(
    input_dim: int,
    width: int = 128,
    dropout_rate: float = 0.2,
    l1_reg: float = 1e-5,
    l2_reg: float = 1e-4,
    name: str = "small_residual_mlp",
) -> keras.Model:
    """Smaller residual multilayer perceptron: width→width→width→64→32→sigmoid.

    Used for low-dimensional inputs (batch GEX, VJ genes).
    """
    reg = l1_l2(l1=l1_reg, l2=l2_reg)
    inp = Input(shape=(input_dim,))

    x = Dense(width, kernel_regularizer=reg)(inp)
    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(dropout_rate)(x)

    res1 = x
    x = Dense(width, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Add()([x, res1]); x = Dropout(dropout_rate)(x)

    res2 = x
    x = Dense(width, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Activation("relu")(x)
    x = Add()([x, res2]); x = Dropout(dropout_rate)(x)

    x = Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Dropout(dropout_rate)(x)

    x = Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x = BatchNormalization()(x); x = Dropout(dropout_rate / 2)(x)

    out = Dense(1, activation="sigmoid", name="output")(x)
    return Model(inp, out, name=name)


def compile_and_train(
    model,
    X_train, y_train,
    X_val,   y_val,
    lr: float = 1e-4,
    batch_size: int = 128,
    epochs: int = 75,
    patience: int = 10,
    weights_path: str = None,
):
    """Compile and train a binary classifier.

    Monitors val_auc_pr with EarlyStopping + ReduceLROnPlateau.
    Optionally saves weights to weights_path.
    Returns Keras History.
    """
    model.compile(
        optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
        ],
    )
    callbacks = [
        EarlyStopping(monitor="val_auc_pr", patience=patience,
                      mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc_pr", factor=0.5,
                          patience=max(3, patience // 3), mode="max", min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    if weights_path:
        model.save_weights(weights_path)
        print(f"Saved weights → {weights_path}")
    return history


def evaluate_classifier(model, X_test, y_test) -> dict:
    """Evaluate a trained classifier on the test set.

    Prints classification report and returns {'roc_auc': ..., 'pr_auc': ...}.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()
    print(classification_report(y_test, y_pred > 0.5,
                                target_names=["Negative (0)", "Positive (1)"]))
    roc_auc = roc_auc_score(y_test, y_pred)
    pr_auc  = average_precision_score(y_test, y_pred)
    print(f"ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def plot_training_curves(history) -> None:
    """Plot loss and AUC-PR training/validation curves side by side."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["auc_pr"], label="Training AUC-PR")
    plt.plot(history.history["val_auc_pr"], label="Validation AUC-PR")
    plt.title("Model AUC-PR"); plt.xlabel("Epoch"); plt.ylabel("AUC-PR")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Sklearn baselines
# =============================================================================

def train_eval_logreg(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Logistic regression (L2, balanced) trained on train+val, evaluated on test."""
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    logreg = LogisticRegression(penalty="l2", class_weight="balanced",
                                max_iter=1000, n_jobs=-1)
    logreg.fit(X_tv, y_tv)

    y_pred   = logreg.predict(X_test)
    y_proba  = logreg.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Neg", "Pos"]))
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    print(f"ROC AUC: {roc_auc:.4f}  |  AUPR: {pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def train_eval_rf(X_train, y_train, X_val, y_val, X_test, y_test,
                  n_estimators: int = 700) -> dict:
    """Random forest trained on train+val.

    Threshold is selected by maximizing F1 on val; final metrics on test.
    """
    X_tv = np.concatenate([X_train, X_val], axis=0)
    y_tv = np.concatenate([y_train, y_val], axis=0)

    rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tv, y_tv)

    val_proba = rf.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[f1.argmax()]

    test_proba   = rf.predict_proba(X_test)[:, 1]
    test_pred    = (test_proba >= best_thr).astype(int)

    print(classification_report(y_test, test_pred, target_names=["Neg", "Pos"]))
    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc  = average_precision_score(y_test, test_proba)
    print(f"ROC AUC: {roc_auc:.4f}  |  AUPR: {pr_auc:.4f}  |  threshold={best_thr:.3f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "threshold": best_thr}


# =============================================================================
# Performance utility
# =============================================================================

def get_basic_run_metrics(model, history, x_example,
                          repeats: int = 50, warmup: int = 10,
                          final_window: int = 10) -> dict:
    """Return param count, latency (ms), best and tail val_loss statistics."""
    val_loss = np.array(history.history["val_loss"], dtype=float)
    window   = min(final_window, len(val_loss))
    val_tail = val_loss[-window:]

    for _ in range(warmup):
        model(x_example, training=False)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x_example, training=False)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "total_params":   round(model.count_params(), 4),
        "latency_ms":     round(float(np.mean(times)), 4),
        "best_val_loss":  round(float(np.min(val_loss)), 4),
        "val_loss_std":   round(float(np.std(val_tail)), 4),
    }

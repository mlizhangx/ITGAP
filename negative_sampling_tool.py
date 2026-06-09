#!/usr/bin/env python3
"""
Negative Sampling Tool for TCR-Peptide Recognition

Generates synthetic negative TCR-peptide pairs from the 10x Genomics
benchmark dataset and combines them with positive pairs into a labeled
CSV ready for model training.
"""

import numpy as np
import pandas as pd
import anndata as ad
import os
import json
import pickle
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NegativeSamplingTool:
    """Generate negative samples and create combined positive+negative datasets."""

    def __init__(self,
                 data_dir: str = "data",
                 output_dir: str = "output/processed_data",
                 negative_ratio: float = 3.0,
                 experiment_name: str = None,
                 random_seed: int = 42,
                 shared_negative_dir: str = "output/negative_samples"):
        """
        Parameters
        ----------
        data_dir : str
            Directory containing the h5ad file.
        output_dir : str
            Directory to save processed datasets.
        negative_ratio : float
            Ratio of negative to positive samples.
        experiment_name : str, optional
            Name for experiment directory (auto-generated if None).
        random_seed : int
            Random seed for reproducibility.
        shared_negative_dir : str
            Directory to save the large negative pool (shared across ratios).
        """
        self.data_dir = Path(data_dir)
        self.negative_ratio = negative_ratio
        self.random_seed = random_seed

        if experiment_name is None:
            if negative_ratio >= 1.0:
                ratio_str = f"neg_ratio_{int(negative_ratio)}_1"
            else:
                inverse_ratio = 1.0 / negative_ratio
                ratio_str = f"neg_ratio_1_{int(inverse_ratio)}"
            experiment_name = ratio_str

        self.experiment_name = experiment_name

        if "experiments" not in str(output_dir):
            self.output_dir = Path(f"output/experiments/{experiment_name}/processed_data")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indices_dir = self.output_dir.parent / "sample_indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        self.shared_negative_dir = Path(shared_negative_dir)
        self.shared_negative_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(random_seed)

        print(f"Negative Sampling Tool initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Experiment: {self.experiment_name}")
        print(f"   Negative ratio: {self.negative_ratio}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Indices directory: {self.indices_dir}")
        print(f"   Shared negative directory: {self.shared_negative_dir}")
        print(f"   Random seed: {self.random_seed}")

    def load_datasets(self) -> ad.AnnData:
        """Load the 10x Genomics benchmark dataset."""
        print("\n Loading dataset...")

        h5ad_10x = self.data_dir / "merge_gex_all_donors_all_peptides_meta_for_Leah_dat.h5ad"
        if not h5ad_10x.exists():
            raise FileNotFoundError(f"10X dataset not found: {h5ad_10x}")

        adata_10x = ad.read_h5ad(h5ad_10x)
        print(f"    10X dataset loaded: {adata_10x.shape}")
        return adata_10x

    def auto_detect_columns(self, adata: ad.AnnData, dataset_name: str) -> Tuple[str, str]:
        """Auto-detect TCR and peptide column names in an AnnData obs."""
        print(f"\n Auto-detecting columns for {dataset_name}...")

        tcr_candidates = ['cdr3_TRB', 'cdr3b', 'cdr3_beta', 'cdr3_trb']
        tcr_col = None
        for col in tcr_candidates:
            if col in adata.obs.columns:
                tcr_col = col
                break

        pep_candidates = ['Peptide_sequence', 'peptide', 'Epitope_ID', 'epitope']
        pep_col = None
        for col in pep_candidates:
            if col in adata.obs.columns:
                pep_col = col
                break

        if tcr_col is None:
            raise ValueError(f"No TCR column found in {dataset_name}. Available: {list(adata.obs.columns)}")
        if pep_col is None:
            raise ValueError(f"No peptide column found in {dataset_name}. Available: {list(adata.obs.columns)}")

        print(f"    TCR column: {tcr_col}")
        print(f"    Peptide column: {pep_col}")
        return tcr_col, pep_col

    def extract_positive_pairs(self, adata: ad.AnnData, tcr_col: str, pep_col: str,
                               dataset_name: str, filter_nas: bool = True) -> pd.DataFrame:
        """
        Extract TCR-peptide pairs from an AnnData object.

        Parameters
        ----------
        filter_nas : bool
            If True, remove rows where either sequence is empty (binding instances only).
            If False, keep all rows for use as a TCR sampling pool.
        """
        print(f"\n Extracting positive pairs from {dataset_name}...")
        print(f"    Original dataset size: {len(adata.obs)} samples")

        tcr_sequences = adata.obs[tcr_col].astype(str).replace('nan', '').values
        peptide_sequences = adata.obs[pep_col].astype(str).replace('nan', '').values
        original_indices = np.arange(len(adata.obs))

        donor_info = None
        if 'donor' in adata.obs.columns:
            donor_info = adata.obs['donor'].astype(str).values
            print(f"    Found donor information")

        data_dict = {
            'tcr': tcr_sequences,
            'peptide': peptide_sequences,
            'label': np.ones(len(tcr_sequences), dtype=int),
            'tcr_source_dataset': [dataset_name] * len(tcr_sequences),
            'tcr_source_index': original_indices,
            'peptide_source_dataset': [dataset_name] * len(tcr_sequences),
            'peptide_source_index': original_indices,
        }

        if donor_info is not None:
            data_dict['donor'] = donor_info

        positive_pairs = pd.DataFrame(data_dict)

        print(f"    Before filtering: {len(positive_pairs)} samples")
        print(f"    Unique TCRs before filtering: {positive_pairs['tcr'].nunique()}")
        print(f"    Unique Peptides before filtering: {positive_pairs['peptide'].nunique()}")

        if filter_nas:
            valid_mask = (positive_pairs['tcr'].str.len() > 0) & (positive_pairs['peptide'].str.len() > 0)
            positive_pairs = positive_pairs[valid_mask].copy()
            print(f"    After filtering: {len(positive_pairs)} valid pairs")
            print(f"    Unique TCRs: {positive_pairs['tcr'].nunique()}")
            print(f"    Unique Peptides: {positive_pairs['peptide'].nunique()}")
        else:
            print(f"    Skipping NA filtering - keeping all {len(positive_pairs)} samples")

        print(f"    tcr_source_index range: {positive_pairs['tcr_source_index'].min()}-{positive_pairs['tcr_source_index'].max()}")
        print(f"    peptide_source_index range: {positive_pairs['peptide_source_index'].min()}-{positive_pairs['peptide_source_index'].max()}")

        return positive_pairs

    def load_large_negative_population(self) -> Optional[pd.DataFrame]:
        """Load existing large negative pool from disk if available."""
        large_pop_path = self.shared_negative_dir / "large_negative_population.csv"
        large_pop_stats_path = self.shared_negative_dir / "large_negative_population_stats.json"

        if large_pop_path.exists() and large_pop_stats_path.exists():
            print(f"\n Loading existing large negative population...")
            large_pop_df = pd.read_csv(large_pop_path)

            with open(large_pop_stats_path, 'r') as f:
                stats = json.load(f)

            print(f"    Loaded {len(large_pop_df):,} negative samples")
            print(f"    k_multiplier: {stats.get('k_multiplier', 'unknown')}")
            print(f"    n_valid_positives: {stats.get('n_valid_positives', 'unknown'):,}")
            print(f"    n_all_tcrs: {stats.get('n_all_tcrs', 'unknown'):,}")
            print(f"    Random seed: {stats.get('random_seed', 'unknown')}")
            print(f"    Unique TCRs: {large_pop_df['tcr'].nunique():,}")
            print(f"    Unique peptides: {large_pop_df['peptide'].nunique():,}")
            print(f"    Binding TCRs (Y): {(large_pop_df['binding_tcr'] == 'Y').sum():,}")
            print(f"    Non-binding TCRs (N): {(large_pop_df['binding_tcr'] == 'N').sum():,}")

            return large_pop_df
        else:
            print(f"\n No existing large negative population found at: {large_pop_path}")
            return None

    def generate_large_negative_population(self, positive_pairs: pd.DataFrame,
                                           all_tcrs: pd.DataFrame,
                                           k_multiplier: int = 20,
                                           force_regenerate: bool = False) -> pd.DataFrame:
        """
        Generate a large pool of negative pairs and save it to the shared directory.

        The pool is generated once and reused across different negative ratios.
        Each TCR in `all_tcrs` is paired with `k_multiplier` randomly sampled
        peptides (uniform over unique peptides), excluding known positive pairs.

        Parameters
        ----------
        positive_pairs : pd.DataFrame
            Valid positive pairs (filter_nas=True).
        all_tcrs : pd.DataFrame
            All TCRs including those without a peptide match (filter_nas=False).
        k_multiplier : int
            Number of negative pairings attempted per TCR.
        force_regenerate : bool
            If True, regenerate even if an existing pool is found on disk.
        """
        if not force_regenerate:
            existing_pop = self.load_large_negative_population()
            if existing_pop is not None:
                print(f"    Using existing large negative population")
                return existing_pop

        print(f"\n{'='*60}")
        print(f" GENERATING LARGE NEGATIVE POPULATION")
        print(f"{'='*60}")
        print(f"    Saving to: {self.shared_negative_dir}")

        valid_mask = (all_tcrs['peptide'].str.len() > 0)
        n_valid_positives = len(positive_pairs)
        n_all_tcrs = len(all_tcrs)
        n_nonvalid_tcrs = (~valid_mask).sum()
        large_population_size = n_all_tcrs * k_multiplier

        print(f"\n    Valid positive TCRs: {n_valid_positives:,}")
        print(f"    Non-valid TCRs (no peptide): {n_nonvalid_tcrs:,}")
        print(f"    Total TCR pool: {n_all_tcrs:,}")
        print(f"    Large population target: {large_population_size:,}")

        # Build uniform peptide pool from unique peptides
        print(f"    Creating uniform peptide pool from unique peptides...")
        peptide_dict = {}
        for _, row in positive_pairs.iterrows():
            pep_seq = row['peptide']
            if pep_seq not in peptide_dict:
                peptide_dict[pep_seq] = {
                    'sequence': pep_seq,
                    'source_dataset': row['peptide_source_dataset'],
                    'source_embedding_index': row['peptide_source_index'],
                    'donor': row.get('donor', 'unknown')
                }
        peptide_instances = list(peptide_dict.values())
        print(f"    Unique peptides available for sampling: {len(peptide_instances):,}")

        positive_set = set(zip(positive_pairs['tcr'], positive_pairs['peptide']))
        print(f"    Unique positive pairs for collision checking: {len(positive_set):,}")

        # Build TCR pool from all_tcrs, marking binding status
        print(f"\n Creating TCR pool from all TCRs...")
        tcr_pool = []
        for _, row in all_tcrs.iterrows():
            tcr_seq = row['tcr']
            is_binding_instance = row.get('instance_type', 'binding') == 'binding'
            tcr_pool.append({
                'sequence': tcr_seq,
                'source_dataset': row['tcr_source_dataset'],
                'source_embedding_index': row['tcr_source_index'],
                'donor': row.get('donor', 'unknown'),
                'is_binding_tcr': is_binding_instance
            })

        binding_tcrs = [t for t in tcr_pool if t['is_binding_tcr']]
        non_binding_tcrs = [t for t in tcr_pool if not t['is_binding_tcr']]

        print(f"    TCR pool size: {len(tcr_pool):,}")
        print(f"    Binding TCRs: {len(binding_tcrs):,}")
        print(f"    Non-binding TCRs: {len(non_binding_tcrs):,}")

        # Generate negatives: k_multiplier pairings per TCR
        print(f"\n Generating large negative population...")
        large_negative_population = []
        negative_pair_set = set()
        total_attempts = 0
        collisions_binding = 0
        collisions_nonbinding = 0
        successes_binding = 0
        successes_nonbinding = 0

        for tcr_idx, tcr_info in enumerate(tcr_pool):
            tcr_seq = tcr_info['sequence']
            is_binding = tcr_info['is_binding_tcr']

            for _ in range(k_multiplier):
                peptide_instance = peptide_instances[np.random.randint(len(peptide_instances))]
                peptide_seq = peptide_instance['sequence']
                pair = (tcr_seq, peptide_seq)

                if pair not in positive_set and pair not in negative_pair_set:
                    large_negative_population.append({
                        'tcr': tcr_seq,
                        'peptide': peptide_seq,
                        'label': 0,
                        'tcr_source_dataset': tcr_info['source_dataset'],
                        'tcr_source_index': tcr_info['source_embedding_index'],
                        'peptide_source_dataset': peptide_instance['source_dataset'],
                        'peptide_source_index': peptide_instance['source_embedding_index'],
                        'binding_tcr': 'Y' if is_binding else 'N'
                    })
                    negative_pair_set.add(pair)
                    if is_binding:
                        successes_binding += 1
                    else:
                        successes_nonbinding += 1
                else:
                    if is_binding:
                        collisions_binding += 1
                    else:
                        collisions_nonbinding += 1

                total_attempts += 1

            if (tcr_idx + 1) % 10000 == 0:
                print(f"    Processed {tcr_idx + 1:,}/{len(tcr_pool):,} TCRs...")

        print(f"    Large population generated: {len(large_negative_population):,} negatives")
        print(f"    Total sampling attempts: {total_attempts:,}")
        print(f"    Collision analysis:")
        print(f"      Binding TCRs: {successes_binding:,} successes, {collisions_binding:,} collisions")
        print(f"      Non-binding TCRs: {successes_nonbinding:,} successes, {collisions_nonbinding:,} collisions")

        # Save large population
        large_pop_df = pd.DataFrame(large_negative_population)
        large_pop_path = self.shared_negative_dir / "large_negative_population.csv"
        large_pop_df.to_csv(large_pop_path, index=False)
        print(f"\n    Large negative population saved: {large_pop_path}")

        large_pop_stats = {
            'total_samples': len(large_negative_population),
            'unique_tcrs': large_pop_df['tcr'].nunique(),
            'unique_peptides': large_pop_df['peptide'].nunique(),
            'tcr_source_10x': int((large_pop_df['tcr_source_dataset'] == '10X').sum()),
            'binding_tcr_yes': int((large_pop_df['binding_tcr'] == 'Y').sum()),
            'binding_tcr_no': int((large_pop_df['binding_tcr'] == 'N').sum()),
            'k_multiplier': k_multiplier,
            'n_valid_positives': n_valid_positives,
            'n_all_tcrs': n_all_tcrs,
            'random_seed': self.random_seed
        }
        large_pop_stats_path = self.shared_negative_dir / "large_negative_population_stats.json"
        with open(large_pop_stats_path, 'w') as f:
            json.dump(large_pop_stats, f, indent=2)
        print(f"    Large population statistics saved: {large_pop_stats_path}")

        return large_pop_df

    def sample_from_large_population(self, large_population: pd.DataFrame,
                                     n_valid_positives: int,
                                     negative_ratio: float) -> pd.DataFrame:
        """
        Sample the final negative set from the large pool.

        Parameters
        ----------
        large_population : pd.DataFrame
            Pre-generated large negative pool.
        n_valid_positives : int
            Number of valid positive samples (determines target size).
        negative_ratio : float
            Desired ratio of negatives to positives.
        """
        final_negatives_size = int(n_valid_positives * negative_ratio)

        print(f"\n Sampling final negatives from large population...")
        print(f"    Large population size: {len(large_population):,}")
        print(f"    Valid positives: {n_valid_positives:,}")
        print(f"    Negative ratio: {negative_ratio}:1")
        print(f"    Target negatives: {final_negatives_size:,}")

        if len(large_population) < final_negatives_size:
            print(f"    Warning: Large population ({len(large_population):,}) is smaller than target ({final_negatives_size:,})")
            print(f"    Using all available negatives")
            final_negatives_df = large_population
        else:
            final_negatives_df = (
                large_population
                .drop_duplicates(subset=["tcr", "peptide"])
                .sample(n=final_negatives_size, replace=False, random_state=self.random_seed)
                .copy()
            )

        print(f"    Final negatives sampled: {len(final_negatives_df):,}")
        print(f"    Actual negative ratio: {len(final_negatives_df)/n_valid_positives:.2f}:1")
        print(f"    Binding TCRs (Y): {(final_negatives_df['binding_tcr'] == 'Y').sum():,} ({(final_negatives_df['binding_tcr'] == 'Y').sum()/len(final_negatives_df)*100:.1f}%)")
        print(f"    Non-binding TCRs (N): {(final_negatives_df['binding_tcr'] == 'N').sum():,} ({(final_negatives_df['binding_tcr'] == 'N').sum()/len(final_negatives_df)*100:.1f}%)")

        return final_negatives_df

    def generate_negative_samples_per_instance(self, positive_pairs: pd.DataFrame,
                                               all_tcrs: pd.DataFrame,
                                               negative_ratio: float = None,
                                               k_multiplier: int = 20,
                                               use_existing_large_pop: bool = True) -> pd.DataFrame:
        """
        Two-stage negative sampling:

        1. Generate (or load) a large pool: all_tcrs × k_multiplier pairings.
        2. Sample the final negatives: positive_pairs × negative_ratio rows.

        Parameters
        ----------
        positive_pairs : pd.DataFrame
            Valid positive TCR-peptide pairs (filter_nas=True).
        all_tcrs : pd.DataFrame
            All TCRs including those without peptide matches (filter_nas=False).
        negative_ratio : float, optional
            Final ratio of negatives to positives (default: self.negative_ratio).
        k_multiplier : int
            Multiplier for the large pool generation step.
        use_existing_large_pop : bool
            If True, load an existing pool from disk rather than regenerating.
        """
        if negative_ratio is None:
            negative_ratio = self.negative_ratio

        n_valid_positives = len(positive_pairs)

        if use_existing_large_pop:
            large_pop_df = self.load_large_negative_population()
            if large_pop_df is None:
                large_pop_df = self.generate_large_negative_population(
                    positive_pairs, all_tcrs, k_multiplier, force_regenerate=False
                )
        else:
            large_pop_df = self.generate_large_negative_population(
                positive_pairs, all_tcrs, k_multiplier, force_regenerate=True
            )

        negative_df = self.sample_from_large_population(
            large_pop_df, n_valid_positives, negative_ratio
        )

        return negative_df

    def create_combined_dataset(self,
                                negative_ratio: float = None,
                                save_mappings: bool = True,
                                k_multiplier: int = 20) -> Dict:
        """
        Build the full positive+negative dataset from the 10x benchmark.

        Loads the h5ad file, extracts positive pairs, generates negatives via
        two-stage per-instance sampling, and saves a combined CSV.

        Parameters
        ----------
        negative_ratio : float, optional
            Ratio of negatives to positives (default: self.negative_ratio).
        save_mappings : bool
            If True, save sequence-to-index mappings as a pickle file.
        k_multiplier : int
            Passed through to generate_large_negative_population.

        Returns
        -------
        dict with keys: 'dataset', 'mappings', 'statistics'
        """
        if negative_ratio is None:
            negative_ratio = self.negative_ratio

        print(f"\n{'='*60}")
        print(f" CREATING COMBINED DATASET (10x Only)")
        print(f"{'='*60}")

        adata_10x = self.load_datasets()
        tcr_col_10x, pep_col_10x = self.auto_detect_columns(adata_10x, "10X")

        # Binding instances: TCRs paired with a valid peptide
        positive_10x = self.extract_positive_pairs(adata_10x, tcr_col_10x, pep_col_10x, "10X", filter_nas=True)

        # Full TCR pool: all cells including those without a peptide annotation
        all_10x_tcrs = self.extract_positive_pairs(adata_10x, tcr_col_10x, pep_col_10x, "10X", filter_nas=False)
        all_10x_tcrs['instance_type'] = all_10x_tcrs['peptide'].apply(
            lambda x: 'binding' if len(str(x)) > 0 else 'non-binding'
        )

        print(f"\n Instance type distribution in TCR pool:")
        print(f"   Binding instances: {(all_10x_tcrs['instance_type'] == 'binding').sum():,}")
        print(f"   Non-binding instances: {(all_10x_tcrs['instance_type'] == 'non-binding').sum():,}")

        all_positive = positive_10x.copy()
        all_positive['binding_tcr'] = 'Y'
        print(f"\n Combined valid positive pairs: {len(all_positive):,}")
        print(f" Combined TCR pool (valid + non-valid): {len(all_10x_tcrs):,}")

        negative_pairs = self.generate_negative_samples_per_instance(
            all_positive, all_10x_tcrs, negative_ratio, k_multiplier
        )

        if save_mappings:
            mappings = self.create_sequence_mappings(positive_10x, negative_pairs)
        else:
            mappings = {}

        combined_dataset = pd.concat([all_positive, negative_pairs], ignore_index=True)

        # Enforce exact ratio
        n_pos = len(all_positive)
        target_neg = int(round(n_pos * negative_ratio))

        negative_subset = (
            combined_dataset[combined_dataset.label == 0]
            .drop_duplicates(["tcr", "peptide"])
            .sample(n=target_neg, random_state=self.random_seed)
        )

        combined_dataset = pd.concat([all_positive, negative_subset], ignore_index=True)

        print(f"\n Final combined dataset: {len(combined_dataset):,} samples")
        print(f"    Positive: {np.sum(combined_dataset['label'] == 1):,} ({np.mean(combined_dataset['label'])*100:.1f}%)")
        print(f"    Negative: {np.sum(combined_dataset['label'] == 0):,} ({(1-np.mean(combined_dataset['label']))*100:.1f}%)")
        actual_ratio = (combined_dataset.label == 0).sum() / (combined_dataset.label == 1).sum()
        print(f"    Exact ratio achieved: {actual_ratio:.4f}")

        self.save_combined_dataset(combined_dataset, mappings)

        return {
            'dataset': combined_dataset,
            'mappings': mappings,
            'statistics': self.calculate_statistics(combined_dataset)
        }

    def create_sequence_mappings(self, positive_10x: pd.DataFrame,
                                 negative_pairs: pd.DataFrame) -> Dict:
        """Create mappings from (tcr, peptide) sequence pairs to source embedding indices."""
        print(f"\n  Creating sequence mappings...")

        tcr_pep_to_idx = {}
        idx_to_tcr_pep = {}

        for _, row in positive_10x.iterrows():
            key = (row['tcr'], row['peptide'])
            tcr_pep_to_idx[key] = row['tcr_source_index']

        idx_to_tcr_pep = {v: k for k, v in tcr_pep_to_idx.items()}

        tcr_to_datasets = {}
        peptide_to_datasets = {}

        for _, row in positive_10x.iterrows():
            tcr = row['tcr']
            peptide = row['peptide']
            idx = row['tcr_source_index']

            if tcr not in tcr_to_datasets:
                tcr_to_datasets[tcr] = []
            tcr_to_datasets[tcr].append({'dataset': '10X', 'index': idx})

            if peptide not in peptide_to_datasets:
                peptide_to_datasets[peptide] = []
            peptide_to_datasets[peptide].append({'dataset': '10X', 'index': idx})

        mappings = {
            'tcr_pep_to_idx': tcr_pep_to_idx,
            'idx_to_tcr_pep': idx_to_tcr_pep,
            'tcr_to_datasets': tcr_to_datasets,
            'peptide_to_datasets': peptide_to_datasets
        }

        print(f"    Created mappings for {len(tcr_pep_to_idx)} pairs")
        print(f"    Unique TCRs: {len(tcr_to_datasets)}")
        print(f"    Unique peptides: {len(peptide_to_datasets)}")

        return mappings

    def calculate_statistics(self, dataset: pd.DataFrame) -> Dict:
        """Calculate summary statistics for a combined dataset."""
        stats = {
            'total_samples': len(dataset),
            'positive_samples': int(np.sum(dataset['label'] == 1)),
            'negative_samples': int(np.sum(dataset['label'] == 0)),
            'unique_tcrs': int(dataset['tcr'].nunique()),
            'unique_peptides': int(dataset['peptide'].nunique()),
            'class_balance': float(np.mean(dataset['label'])),
            'negative_ratio': float(np.sum(dataset['label'] == 0) / np.sum(dataset['label'] == 1))
        }
        return stats

    def save_combined_dataset(self, dataset: pd.DataFrame, mappings: Dict):
        """Save dataset CSV, sequence mappings, statistics JSON, and a README."""
        print(f"\n Saving combined dataset...")

        dataset_path = self.output_dir / "combined_tcr_peptide_dataset.csv"
        dataset.to_csv(dataset_path, index=False)
        print(f"    Dataset saved: {dataset_path}")

        if mappings:
            mappings_path = self.output_dir / "sequence_mappings.pkl"
            with open(mappings_path, 'wb') as f:
                pickle.dump(mappings, f)
            print(f"    Mappings saved: {mappings_path}")

        stats = self.calculate_statistics(dataset)
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"    Statistics saved: {stats_path}")

        readme_content = self.create_readme(dataset, stats)
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"    README saved: {readme_path}")

    def create_readme(self, dataset: pd.DataFrame, stats: Dict) -> str:
        """Generate a README documenting the saved dataset files."""
        readme = f"""# TCR-Peptide Combined Dataset

## Overview
Positive and negative TCR-peptide pairs for training TCR-antigen recognition models.

## Dataset Statistics
- **Total Samples**: {stats['total_samples']:,}
- **Positive Samples**: {stats['positive_samples']:,} ({stats['class_balance']*100:.1f}%)
- **Negative Samples**: {stats['negative_samples']:,} ({(1-stats['class_balance'])*100:.1f}%)
- **Unique TCRs**: {stats['unique_tcrs']:,}
- **Unique Peptides**: {stats['unique_peptides']:,}
- **Negative Ratio**: {stats['negative_ratio']:.2f}:1

## Data Sources
- **10x Genomics benchmark** — positive TCR-peptide pairs from CD8+ T cell multi-omics data
- **Synthetic negatives** — generated by pairing real TCRs with randomly sampled peptides

## Files
- `combined_tcr_peptide_dataset.csv`: Main dataset
- `sequence_mappings.pkl`: Mappings from (tcr, peptide) sequences to source embedding indices
- `dataset_statistics.json`: Summary statistics
- `README.md`: This file

## Usage
```python
import pandas as pd
import pickle

dataset = pd.read_csv('combined_tcr_peptide_dataset.csv')

with open('sequence_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

positive_pairs = dataset[dataset['label'] == 1]
negative_pairs = dataset[dataset['label'] == 0]
```

## Columns
| Column | Description |
|---|---|
| `tcr` | CDR3β sequence |
| `peptide` | Peptide sequence |
| `label` | 1 = positive (known binding), 0 = negative (synthetic) |
| `tcr_source_dataset` | Source dataset for the TCR (10X) |
| `tcr_source_index` | Row index in the original AnnData object |
| `peptide_source_dataset` | Source dataset for the peptide (10X) |
| `peptide_source_index` | Row index in the original AnnData object |
| `binding_tcr` | Y if this TCR has at least one known binding partner |

## Negative Sampling Strategy
Negatives are generated in two stages:
1. **Large pool**: For each TCR in the dataset (including TCRs without a known peptide match),
   `k_multiplier` candidate peptides are drawn uniformly at random from the set of
   unique positive peptides. Known positive pairs are excluded.
2. **Final sample**: `n_positives × negative_ratio` rows are sampled without replacement
   from the pool to produce the final dataset.

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return readme


def main():
    """Generate negative samples and combined dataset for the 10x benchmark."""
    print("TCR-Peptide Negative Sampling Tool")
    print("=" * 50)

    tool = NegativeSamplingTool(
        data_dir="data",
        negative_ratio=3.0,
        random_seed=42
    )

    result = tool.create_combined_dataset(negative_ratio=3.0)

    print(f"\nProcess completed successfully!")
    print(f"Final dataset shape: {result['dataset'].shape}")
    print(f"Files saved to: {tool.output_dir}")


if __name__ == "__main__":
    main()

"""
Phase 2: Temporal Dynamics of State-Gating
==========================================
Investigates whether state-gating emerges gradually or switches abruptly
along differentiation trajectories. Uses diffusion pseudotime from the
Tabula Sapiens immune data and sliding-window correlation analysis.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import sparse, stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, time
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "/Users/ihorkendiukhov/biodyn-work/single_cell_mechinterp/data/raw"
OUT_DIR = "/Users/ihorkendiukhov/biodyn-work/subproject_28_cell_state_conditional_circuits"
FIG_DIR = os.path.join(OUT_DIR, "reports", "figures")
IMPL_DIR = os.path.join(OUT_DIR, "implementation")
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
print("Loading data...")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "tabula_sapiens_immune_subset_20000.h5ad"))
adata.obs['regime'] = adata.obs['cell_type'].astype(str)
regime_counts = adata.obs['regime'].value_counts()
valid_regimes = regime_counts[regime_counts >= 100].index.tolist()
adata = adata[adata.obs['regime'].isin(valid_regimes)].copy()
print(f"Data: {adata.shape[0]} cells, {len(valid_regimes)} regimes")

# Gene mapping
symbol_to_ensembl = {}
ensembl_to_symbol = {}
for ens_id, row in adata.var.iterrows():
    sym = row['feature_name']
    if pd.notna(sym):
        symbol_to_ensembl[sym] = ens_id
        ensembl_to_symbol[ens_id] = sym


def spearman_safe(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    return (r, p) if not np.isnan(r) else (0.0, 1.0)


# ============================================================
# COMPUTE PSEUDOTIME VIA DIFFUSION PSEUDOTIME
# ============================================================
print("\nComputing pseudotime trajectories...")

# Use highly variable genes for dimensionality reduction
adata_proc = adata.copy()

# Normalize if needed
if adata_proc.X.max() > 50:
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

sc.pp.highly_variable_genes(adata_proc, n_top_genes=2000)
sc.pp.pca(adata_proc, n_comps=30)
sc.pp.neighbors(adata_proc, n_neighbors=15, n_pcs=30)

# Diffusion map for pseudotime
sc.tl.diffmap(adata_proc, n_comps=15)

# Define multiple root cells for different lineages
# We pick representative root cells from progenitor/naive populations
lineage_trajectories = {
    'T_cell_trajectory': {
        'include_regimes': [
            'T cell',
            'CD4-positive, alpha-beta T cell',
            'CD8-positive, alpha-beta T cell',
            'CD4-positive, alpha-beta thymocyte',
            'naive thymus-derived CD4-positive, alpha-beta T cell',
            'mature NK T cell',
            'regulatory T cell',
            'natural killer cell',
        ],
        'root_regime': 'T cell',  # Naive/progenitor T cells as root
        'edges': [
            ("RUNX3", "CD8A"), ("TCF7", "LEF1"), ("TCF7", "IL7R"),
            ("TBX21", "IFNG"), ("FOXP3", "CTLA4"), ("FOXP3", "IL2RA"),
            ("EOMES", "GZMB"), ("EOMES", "PRF1"),
        ],
    },
    'B_cell_trajectory': {
        'include_regimes': [
            'B cell',
            'plasma cell',
        ],
        'root_regime': 'B cell',
        'edges': [
            ("PAX5", "CD79A"), ("PAX5", "CD19"), ("PAX5", "MS4A1"),
            ("EBF1", "CD79A"), ("EBF1", "CD79B"),
            ("IRF4", "PRDM1"), ("PRDM1", "XBP1"),
        ],
    },
    'Myeloid_trajectory': {
        'include_regimes': [
            'monocyte',
            'classical monocyte',
            'intermediate monocyte',
            'macrophage',
            'mononuclear phagocyte',
            'mast cell',
        ],
        'root_regime': 'monocyte',
        'edges': [
            ("SPI1", "ITGAM"), ("SPI1", "CD68"), ("SPI1", "CSF1R"),
            ("NFKB1", "IL1B"), ("NFKB1", "TNF"), ("NFKB1", "CXCL8"),
            ("CEBPA", "CSF3R"), ("HIF1A", "VEGFA"),
        ],
    },
}

# ============================================================
# SLIDING WINDOW CORRELATION ANALYSIS
# ============================================================
print("\nSliding window correlation along pseudotime...")

WINDOW_SIZE_FRAC = 0.15  # 15% of cells per window
STEP_SIZE_FRAC = 0.05    # 5% step

all_trajectory_results = []
transition_records = []

for traj_name, traj_config in lineage_trajectories.items():
    print(f"\n--- {traj_name} ---")

    # Subset to trajectory cells
    include_regimes = [r for r in traj_config['include_regimes'] if r in valid_regimes]
    if len(include_regimes) < 2:
        print(f"  Skipping: only {len(include_regimes)} valid regimes")
        continue

    traj_mask = adata_proc.obs['regime'].isin(include_regimes)
    n_traj = traj_mask.sum()
    if n_traj < 200:
        print(f"  Skipping: only {n_traj} cells")
        continue

    print(f"  Cells: {n_traj}, Regimes: {include_regimes}")

    # Compute pseudotime for this subset
    adata_traj = adata_proc[traj_mask].copy()
    sc.pp.neighbors(adata_traj, n_neighbors=15, n_pcs=30)
    sc.tl.diffmap(adata_traj, n_comps=10)

    # Find root cell in the root regime
    root_regime = traj_config['root_regime']
    if root_regime in include_regimes:
        root_candidates = np.where(adata_traj.obs['regime'] == root_regime)[0]
        if len(root_candidates) > 0:
            # Use diffmap component 1 to pick root
            dc1 = adata_traj.obsm['X_diffmap'][:, 0]
            # Pick the cell in root regime with extreme DC1 value
            root_idx = root_candidates[np.argmin(dc1[root_candidates])]
        else:
            root_idx = 0
    else:
        root_idx = 0

    adata_traj.uns['iroot'] = root_idx

    try:
        sc.tl.dpt(adata_traj)
        pseudotime = adata_traj.obs['dpt_pseudotime'].values
        # Handle infinite values
        valid_pt = np.isfinite(pseudotime)
        if valid_pt.sum() < 100:
            print(f"  Skipping: only {valid_pt.sum()} finite pseudotime values")
            continue
        pseudotime[~valid_pt] = np.nanmax(pseudotime[valid_pt])
    except Exception as e:
        print(f"  DPT failed: {e}. Using DC1 as proxy.")
        pseudotime = adata_traj.obsm['X_diffmap'][:, 0]
        # Flip if needed so root regime has low pseudotime
        root_mask = adata_traj.obs['regime'] == root_regime
        if root_mask.sum() > 0 and np.mean(pseudotime[root_mask]) > np.mean(pseudotime):
            pseudotime = -pseudotime
        # Normalize to [0, 1]
        pseudotime = (pseudotime - pseudotime.min()) / max(pseudotime.max() - pseudotime.min(), 1e-8)

    # Expression matrix for this trajectory
    X_traj = adata_traj.X
    if sparse.issparse(X_traj):
        X_traj = X_traj.toarray()

    # Sort by pseudotime
    pt_order = np.argsort(pseudotime)
    pseudotime_sorted = pseudotime[pt_order]
    X_sorted = X_traj[pt_order]
    regimes_sorted = adata_traj.obs['regime'].values[pt_order]

    n_cells = len(pt_order)
    window_size = max(int(n_cells * WINDOW_SIZE_FRAC), 50)
    step_size = max(int(n_cells * STEP_SIZE_FRAC), 10)

    print(f"  Pseudotime range: [{pseudotime_sorted[0]:.3f}, {pseudotime_sorted[-1]:.3f}]")
    print(f"  Window size: {window_size}, Step size: {step_size}")

    # Sliding window analysis for each edge
    for tf_sym, tgt_sym in traj_config['edges']:
        tf_ens = symbol_to_ensembl.get(tf_sym)
        tgt_ens = symbol_to_ensembl.get(tgt_sym)
        if tf_ens is None or tgt_ens is None:
            continue

        tf_idx = np.where(adata_traj.var_names == tf_ens)[0]
        tgt_idx = np.where(adata_traj.var_names == tgt_ens)[0]
        if len(tf_idx) == 0 or len(tgt_idx) == 0:
            continue
        tf_idx, tgt_idx = tf_idx[0], tgt_idx[0]

        tf_sorted = X_sorted[:, tf_idx]
        tgt_sorted = X_sorted[:, tgt_idx]

        windows = []
        start = 0
        while start + window_size <= n_cells:
            end = start + window_size
            tf_w = tf_sorted[start:end]
            tgt_w = tgt_sorted[start:end]
            pt_w = pseudotime_sorted[start:end]
            regimes_w = regimes_sorted[start:end]

            rho_w, p_w = spearman_safe(tf_w, tgt_w)

            # Dominant regime in window
            regime_counts_w = pd.Series(regimes_w).value_counts()
            dominant_regime = regime_counts_w.index[0]
            dominant_frac = regime_counts_w.iloc[0] / len(regimes_w)

            windows.append({
                'trajectory': traj_name,
                'tf': tf_sym, 'target': tgt_sym,
                'window_start': start,
                'window_center': (start + end) / 2 / n_cells,
                'pt_mean': np.mean(pt_w),
                'pt_min': pt_w[0], 'pt_max': pt_w[-1],
                'rho': rho_w, 'p_val': p_w,
                'n_cells': window_size,
                'dominant_regime': dominant_regime,
                'dominant_frac': dominant_frac,
                'tf_mean': np.mean(tf_w),
                'tgt_mean': np.mean(tgt_w),
            })

            start += step_size

        if len(windows) < 3:
            continue

        df_windows = pd.DataFrame(windows)
        all_trajectory_results.append(df_windows)

        # Detect transition points: where correlation changes sign or shifts > 0.2
        rhos = df_windows['rho'].values
        for i in range(1, len(rhos)):
            delta = abs(rhos[i] - rhos[i-1])
            if delta > 0.15:  # Sharp transition
                transition_records.append({
                    'trajectory': traj_name,
                    'tf': tf_sym, 'target': tgt_sym,
                    'window_idx': i,
                    'pt_position': df_windows.iloc[i]['pt_mean'],
                    'rho_before': rhos[i-1],
                    'rho_after': rhos[i],
                    'delta_rho': rhos[i] - rhos[i-1],
                    'regime_before': df_windows.iloc[i-1]['dominant_regime'],
                    'regime_after': df_windows.iloc[i]['dominant_regime'],
                })

    # Regime composition along pseudotime
    print(f"\n  Regime composition along pseudotime:")
    n_bins_pt = 5
    pt_bins = np.linspace(pseudotime_sorted[0], pseudotime_sorted[-1], n_bins_pt + 1)
    for b in range(n_bins_pt):
        bin_mask = (pseudotime_sorted >= pt_bins[b]) & (pseudotime_sorted < pt_bins[b+1])
        if bin_mask.sum() == 0:
            continue
        regimes_bin = regimes_sorted[bin_mask]
        counts = pd.Series(regimes_bin).value_counts(normalize=True)
        top = ', '.join([f"{r}: {100*v:.0f}%" for r, v in counts.head(3).items()])
        print(f"    PT [{pt_bins[b]:.2f}-{pt_bins[b+1]:.2f}]: n={bin_mask.sum()}, {top}")


# ============================================================
# RESULTS SUMMARY
# ============================================================
if len(all_trajectory_results) > 0:
    df_traj = pd.concat(all_trajectory_results, ignore_index=True)
    df_transitions = pd.DataFrame(transition_records)

    print(f"\n{'='*60}")
    print("TEMPORAL DYNAMICS RESULTS")
    print(f"{'='*60}")
    print(f"\nTotal sliding windows: {len(df_traj)}")
    print(f"Transition events (|delta rho| > 0.15): {len(df_transitions)}")

    if len(df_transitions) > 0:
        print("\nTransition Events:")
        print(df_transitions.to_string(index=False))

        # Are transitions gradual or abrupt?
        # Measure: max |delta rho| between consecutive windows for each edge
        edge_max_delta = df_traj.groupby(['trajectory', 'tf', 'target']).apply(
            lambda g: g['rho'].diff().abs().max()
        ).reset_index(name='max_step_delta')

        # Compare to total range
        edge_range = df_traj.groupby(['trajectory', 'tf', 'target']).agg(
            rho_range=('rho', lambda x: x.max() - x.min()),
            n_windows=('rho', 'count'),
        ).reset_index()

        edge_dynamics = edge_max_delta.merge(edge_range, on=['trajectory', 'tf', 'target'])
        edge_dynamics['abruptness'] = edge_dynamics['max_step_delta'] / edge_dynamics['rho_range'].clip(lower=0.01)
        # abruptness near 1 = single-step transition; near 0.2 = gradual over 5 steps

        print("\nEdge Dynamics Summary:")
        print(edge_dynamics.sort_values('rho_range', ascending=False).to_string(index=False))

        # Classify: abrupt (>0.5) vs gradual (<=0.5)
        n_abrupt = (edge_dynamics['abruptness'] > 0.5).sum()
        n_gradual = (edge_dynamics['abruptness'] <= 0.5).sum()
        print(f"\n  Abrupt transitions (abruptness > 0.5): {n_abrupt}")
        print(f"  Gradual transitions (abruptness <= 0.5): {n_gradual}")

    # Save results
    df_traj.to_csv(os.path.join(IMPL_DIR, "phase2_temporal_dynamics.csv"), index=False)
    if len(df_transitions) > 0:
        df_transitions.to_csv(os.path.join(IMPL_DIR, "phase2_transition_events.csv"), index=False)

    # ============================================================
    # FIGURES
    # ============================================================
    print("\nGenerating temporal dynamics figures...")

    # Figure P2-6: Sliding window correlation along pseudotime
    n_traj = len(lineage_trajectories)
    fig, axes = plt.subplots(n_traj, 1, figsize=(14, 4 * n_traj))
    if n_traj == 1:
        axes = [axes]

    cmap = plt.cm.tab10
    for ax_idx, (traj_name, _) in enumerate(lineage_trajectories.items()):
        ax = axes[ax_idx]
        traj_data = df_traj[df_traj['trajectory'] == traj_name]
        if len(traj_data) == 0:
            ax.text(0.5, 0.5, f'No data for {traj_name}', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        edges_in_traj = traj_data.groupby(['tf', 'target']).size().reset_index()
        for i, (_, row) in enumerate(edges_in_traj.iterrows()):
            sub = traj_data[(traj_data['tf'] == row['tf']) & (traj_data['target'] == row['target'])]
            ax.plot(sub['pt_mean'], sub['rho'], 'o-', color=cmap(i % 10),
                    label=f"{row['tf']}->{row['target']}", markersize=3, alpha=0.7)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Pseudotime')
        ax.set_ylabel('Spearman rho')
        ax.set_title(f'{traj_name}: Edge Correlation Along Pseudotime')
        ax.legend(fontsize=7, loc='best', ncol=2)

        # Mark transitions
        trans = df_transitions[df_transitions['trajectory'] == traj_name]
        for _, t in trans.iterrows():
            ax.axvline(x=t['pt_position'], color='red', alpha=0.3, linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "phase2_fig6_temporal_dynamics.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure P2-7: Abruptness analysis
    if len(df_transitions) > 0 and 'edge_dynamics' in dir() and len(edge_dynamics) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Distribution of abruptness
        axes[0].hist(edge_dynamics['abruptness'].values, bins=15, color='#2c3e50',
                     alpha=0.7, edgecolor='white')
        axes[0].axvline(x=0.5, color='red', linestyle='--', label='Abrupt/Gradual boundary')
        axes[0].set_xlabel('Abruptness Score')
        axes[0].set_ylabel('Number of edges')
        axes[0].set_title('Distribution of Transition Abruptness')
        axes[0].legend()

        # Panel B: rho range vs abruptness
        colors_traj = {'T_cell_trajectory': '#e74c3c', 'B_cell_trajectory': '#3498db',
                       'Myeloid_trajectory': '#2ecc71'}
        for traj_name_p in edge_dynamics['trajectory'].unique():
            sub = edge_dynamics[edge_dynamics['trajectory'] == traj_name_p]
            axes[1].scatter(sub['rho_range'], sub['abruptness'],
                           c=colors_traj.get(traj_name_p, 'gray'), s=60, alpha=0.7,
                           label=traj_name_p.replace('_', ' '))
            for _, row in sub.iterrows():
                axes[1].annotate(f"{row['tf']}->{row['target']}",
                                (row['rho_range'], row['abruptness']),
                                fontsize=6, alpha=0.7)
        axes[1].set_xlabel('Total rho range along trajectory')
        axes[1].set_ylabel('Abruptness')
        axes[1].set_title('Edge Dynamics: Range vs Abruptness')
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "phase2_fig7_transition_abruptness.png"), dpi=150, bbox_inches='tight')
        plt.close()
else:
    print("\nNo trajectory results to report.")

print("\nPhase 2 temporal dynamics analysis complete!")

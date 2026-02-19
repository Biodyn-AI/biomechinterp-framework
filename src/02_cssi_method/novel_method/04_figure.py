#!/usr/bin/env python3
"""Generate CSSI scaling figure from experiment results."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, 'scaling_v2_results.csv'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for ax, metric, label in [(ax1, 'f1', 'F1 Score'), (ax2, 'auc', 'AUROC')]:
    for method, col, color, marker in [
        ('Pooled', f'pool_{metric}', '#2196F3', 'o'),
        ('CSSI-max', f'cssi_max_{metric}', '#F44336', 's'),
        ('CSSI-mean', f'cssi_mean_{metric}', '#4CAF50', '^'),
    ]:
        means = df.groupby('n_states')[col].mean()
        stds = df.groupby('n_states')[col].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=method, color=color, marker=marker, capsize=3, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Cell States', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('CSSI Mitigates Scaling Failure', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_cssi_scaling.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_cssi_scaling.pdf'), bbox_inches='tight')
print("Figures saved.")

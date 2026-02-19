#!/bin/bash
export TOKENIZERS_PARALLELISM=false
echo "=== STEP 1: CSSI Experiment ==="
python3 /mnt/d/openclaw/biodyn-nmi-paper/cssi_real/cssi_experiment.py 2>&1
echo "=== STEP 2: Remaining 6 Perturbation Genes ==="
bash /mnt/d/openclaw/intelligence-augmentation/analysis/results/insilico_wsl/launch_final6.sh 2>&1
echo "=== ALL SEQUENTIAL TASKS COMPLETE ==="

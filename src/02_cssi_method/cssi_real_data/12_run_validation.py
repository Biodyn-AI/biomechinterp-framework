#!/usr/bin/env python3
"""
CSSI Validation Runner

Simple script to run both cross-dataset and cross-tissue validation experiments.
"""

import sys
from pathlib import Path

# CRITICAL: scGPT Setup - torchtext shim MUST be loaded first
exec(open("/mnt/c/Users/Agent/.openclaw/workspace/patch_torchtext.py").read())

import subprocess
import logging
import argparse
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_cross_dataset_validation():
    """Run the main cross-dataset validation experiment."""
    logger.info("🔬 Starting CSSI Cross-Dataset Validation...")
    logger.info("This addresses the circularity problem in the NMI paper")
    
    script_path = Path(__file__).parent / "09_cross_dataset_validation.py"
    
    # Use WSL to run Python script since we're on Windows
    cmd = [
        "wsl", "-u", "agent", "--", "bash", "-lc",
        f"cd /mnt/d/openclaw/biodyn-nmi-paper/cssi_real && python3 {script_path.name} --batch-size 2"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("✅ Cross-dataset validation completed successfully!")
            print(result.stdout)
        else:
            logger.error("❌ Cross-dataset validation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Cross-dataset validation timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"❌ Cross-dataset validation error: {e}")
        return False
    
    return True


def run_cross_tissue_validation():
    """Run the bonus cross-tissue validation experiment."""
    logger.info("🧬 Starting CSSI Cross-Tissue Validation (BONUS)...")
    logger.info("Testing layer transfer: immune → kidney")
    
    script_path = Path(__file__).parent / "10_cross_tissue_validation.py"
    
    # Use WSL to run Python script
    cmd = [
        "wsl", "-u", "agent", "--", "bash", "-lc",
        f"cd /mnt/d/openclaw/biodyn-nmi-paper/cssi_real && python3 {script_path.name} --source-tissue immune --target-tissue kidney --batch-size 2"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            logger.info("✅ Cross-tissue validation completed successfully!")
            print(result.stdout)
        else:
            logger.error("❌ Cross-tissue validation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Cross-tissue validation timed out after 1 hour")
        return False
    except Exception as e:
        logger.error(f"❌ Cross-tissue validation error: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="CSSI Validation Runner")
    parser.add_argument("--cross-dataset", action="store_true", help="Run cross-dataset validation")
    parser.add_argument("--cross-tissue", action="store_true", help="Run cross-tissue validation")
    parser.add_argument("--both", action="store_true", help="Run both validations")
    
    args = parser.parse_args()
    
    if not any([args.cross_dataset, args.cross_tissue, args.both]):
        # Default: run both
        args.both = True
    
    start_time = time.time()
    success_count = 0
    total_runs = 0
    
    if args.cross_dataset or args.both:
        total_runs += 1
        if run_cross_dataset_validation():
            success_count += 1
        
        # Wait between experiments to avoid GPU memory issues
        if args.both or args.cross_tissue:
            logger.info("⏸️ Waiting 30 seconds between experiments...")
            time.sleep(30)
    
    if args.cross_tissue or args.both:
        total_runs += 1
        if run_cross_tissue_validation():
            success_count += 1
    
    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n📊 VALIDATION SUMMARY")
    logger.info(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"✅ Success rate: {success_count}/{total_runs}")
    
    if success_count == total_runs:
        logger.info("🎉 All validations completed successfully!")
        logger.info("📝 Check the generated reports:")
        logger.info("   - CSSI_CROSS_VALIDATION_REPORT.md")
        if total_runs > 1:
            logger.info("   - CSSI_CROSS_TISSUE_REPORT_immune_to_kidney.md")
    else:
        logger.error("❌ Some validations failed. Check the logs above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
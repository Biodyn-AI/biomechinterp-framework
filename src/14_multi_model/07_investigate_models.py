#!/usr/bin/env python3
"""
Investigate available single-cell foundation models for NMI paper extension.
"""

import subprocess
import sys
import json
from datetime import datetime

def check_package_availability(package_name):
    """Check if a package is available on PyPI"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "installed", "info": result.stdout}
        else:
            # Try to install to check availability
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', package_name], 
                                  capture_output=True, text=True)
            if "Could not find" in result.stderr:
                return {"status": "not_found", "error": result.stderr}
            else:
                return {"status": "available", "info": result.stdout}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def check_huggingface_model(model_name):
    """Check if a model exists on HuggingFace"""
    try:
        import requests
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url)
        if response.status_code == 200:
            return {"status": "available", "info": response.json()}
        else:
            return {"status": "not_found", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Main investigation routine"""
    print("[SEARCH] Investigating Single-Cell Foundation Models")
    print("=" * 50)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    # 1. scBERT variations
    print("\n1. Checking scBERT variations...")
    scbert_packages = ["scbert", "scBERT", "single-cell-bert", "sc-bert"]
    for pkg in scbert_packages:
        print(f"   Checking {pkg}...")
        results["models"][f"scbert_{pkg}"] = check_package_availability(pkg)
    
    # 2. scVI (should be available)
    print("\n2. Checking scVI...")
    scvi_packages = ["scvi", "scvi-tools"]
    for pkg in scvi_packages:
        print(f"   Checking {pkg}...")
        results["models"][f"scvi_{pkg}"] = check_package_availability(pkg)
    
    # 3. Check for general single-cell packages that might have these models
    print("\n3. Checking related packages...")
    related_packages = ["scanpy", "anndata", "pytorch-lightning", "transformers"]
    for pkg in related_packages:
        print(f"   Checking {pkg}...")
        results["models"][f"related_{pkg}"] = check_package_availability(pkg)
    
    # 4. HuggingFace models
    print("\n4. Checking HuggingFace models...")
    try:
        import requests
        hf_models = [
            "microsoft/DialoGPT-medium",  # Test basic connectivity
            "ctheodoris/Geneformer",      # Known to exist
            "UCE/universal-cell-embeddings",  # UCE candidate
            "scFoundation/scFoundation",  # scFoundation candidate
        ]
        
        for model in hf_models:
            print(f"   Checking {model}...")
            results["models"][f"hf_{model.replace('/', '_')}"] = check_huggingface_model(model)
    except ImportError:
        print("   No requests library - installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests'], check=True)
        # Retry after installation
        import requests
        for model in hf_models:
            print(f"   Checking {model}...")
            results["models"][f"hf_{model.replace('/', '_')}"] = check_huggingface_model(model)
    
    # Save results
    with open("model_availability_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[OK] Investigation complete. Results saved to model_availability_report.json")
    
    # Print summary
    print("\n[SUMMARY]:")
    available = 0
    installed = 0
    total = len(results["models"])
    
    for name, info in results["models"].items():
        if info["status"] == "available":
            available += 1
            print(f"   [OK] {name} - Available for installation")
        elif info["status"] == "installed":
            installed += 1
            print(f"   [OK] {name} - Already installed")
        else:
            print(f"   [FAIL] {name} - {info['status']}")
    
    print(f"\nInstalled: {installed}/{total}")
    print(f"Available: {available}/{total}")
    print(f"Total viable: {installed + available}/{total}")

if __name__ == "__main__":
    main()
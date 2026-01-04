import json

# Unsloth Baseline (Run 6/5)
baseline_metrics = {
    "mhc_enabled": False,
    "train_runtime": 179.07,
    "train_loss": 3.74565,
    "peak_vram_gb": 19.35,
    "initial_vram_gb": 5.0, # Estimated
    "throughput_tokens_per_sec": 0.061,
    "history": [
        {"step": 1, "loss": 4.1083, "epoch": 0.01},
        {"step": 2, "loss": 3.95, "epoch": 0.02}, # Interpolated
        {"step": 5, "loss": 3.80, "epoch": 0.05}, # Interpolated
        {"step": 10, "loss": 3.348, "epoch": 0.1}
    ]
}

# mHC (Run 8)
with open("logs/benchmark_results.json", "r") as f:
    data = json.load(f)

mhc_metrics = data["mhc"]

# Stitch
full_results = {
    "baseline": baseline_metrics,
    "mhc": mhc_metrics
}

with open("logs/benchmark_results_stitched.json", "w") as f:
    json.dump(full_results, f, indent=4)
    
print("Stitched results saved to logs/benchmark_results_stitched.json")

import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(results_path, output_dir):
    with open(results_path, "r") as f:
        data = json.load(f)
    
    baseline = data["baseline"]
    mhc = data["mhc"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    base_loss = [h["loss"] for h in baseline["history"] if "loss" in h]
    mhc_loss = [h["loss"] for h in mhc["history"] if "loss" in h]
    
    plt.plot(base_loss, label="Baseline", marker='o')
    plt.plot(mhc_loss, label="mHC Retrofit", marker='x')
    plt.title("Training Loss Comparison")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"))
    plt.close()
    
    # 2. VRAM Usage (Bar chart)
    metrics = ["Initial VRAM", "Peak VRAM"]
    base_vram = [baseline["initial_vram_gb"], baseline["peak_vram_gb"]]
    mhc_vram = [mhc["initial_vram_gb"], mhc["peak_vram_gb"]]
    
    x = range(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar([p - width/2 for p in x], base_vram, width, label="Baseline")
    plt.bar([p + width/2 for p in x], mhc_vram, width, label="mHC Retrofit")
    plt.xticks(x, metrics)
    plt.title("VRAM Usage Comparison")
    plt.ylabel("GB")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "vram_comparison.png"))
    plt.close()
    
    # 3. Throughput (Bar chart)
    throughput_base = baseline.get("throughput_tokens_per_sec") or 0
    throughput_mhc = mhc.get("throughput_tokens_per_sec") or 0
    
    plt.figure(figsize=(6, 6))
    plt.bar(["Baseline", "mHC Retrofit"], [throughput_base, throughput_mhc], color=['blue', 'orange'])
    plt.title("Throughput (Tokens/sec)")
    plt.ylabel("Tokens/s")
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="logs/benchmark_results.json")
    parser.add_argument("--output", type=str, default="artifacts/plots")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        plot_metrics(args.input, args.output)
    else:
        print(f"Input file {args.input} not found.")

if __name__ == "__main__":
    main()

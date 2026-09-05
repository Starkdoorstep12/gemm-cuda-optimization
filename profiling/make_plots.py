"""
Generate the core set of plots for the GEMM optimization report.
"""
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

def load_consolidated():
    rows = []
    with open("profiling/consolidated_results.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["size"] = int(r["size"])
            r["gflops"] = float(r["gflops"])
            r["pct_peak"] = float(r["pct_peak"])
            rows.append(r)
    return rows

KERNEL_ORDER = ["Naive", "Coalesced", "SharedMem", "1DBlocktiled",
                "2DBlocktiled", "BestAutotuned", "Warptiled", "cuBLAS"]
GPU_COLORS = {"L40S": "#2E86AB", "A100": "#E63946", "RTX6000Ada": "#06A77D"}


# --- Plot 1: Kernel progression at 2048 (or nearest available) per GPU ---
def plot_kernel_progression(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gpu in ["L40S", "A100", "RTX6000Ada"]:
        xs, ys = [], []
        for k in KERNEL_ORDER:
            # prefer 2048, fallback to closest available size for that kernel/gpu
            candidates = [r for r in rows if r["kernel"] == k and r["gpu"] == gpu]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: abs(r["size"] - 2048))
            xs.append(k)
            ys.append(best["gflops"])
        ax.plot(xs, ys, marker='o', label=gpu, color=GPU_COLORS[gpu], linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel("GFLOPS (log scale)")
    ax.set_title("Kernel Optimization Progression (~2048³, by architecture)")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("plots/01_kernel_progression.png")
    plt.close()
    print("Saved 01_kernel_progression.png")


# --- Plot 2: % of peak achieved, same data, different lens ---
def plot_pct_peak_progression(rows):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gpu in ["L40S", "A100", "RTX6000Ada"]:
        xs, ys = [], []
        for k in KERNEL_ORDER:
            candidates = [r for r in rows if r["kernel"] == k and r["gpu"] == gpu]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: abs(r["size"] - 2048))
            xs.append(k)
            ys.append(best["pct_peak"])
        ax.plot(xs, ys, marker='s', label=gpu, color=GPU_COLORS[gpu], linewidth=2)
    ax.set_ylabel("% of architecture's peak FP32")
    ax.set_title("Fraction of Peak FP32 Achieved by Kernel Stage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("plots/02_pct_of_peak_progression.png")
    plt.close()
    print("Saved 02_pct_of_peak_progression.png")


# --- Plot 3: Naive kernel non-monotonic scaling with matrix size ---
def plot_naive_scaling(rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    for gpu in ["L40S", "A100", "RTX6000Ada"]:
        pts = sorted([r for r in rows if r["kernel"] == "Naive" and r["gpu"] == gpu],
                     key=lambda r: r["size"])
        xs = [r["size"] for r in pts]
        ys = [r["gflops"] for r in pts]
        ax.plot(xs, ys, marker='o', label=gpu, color=GPU_COLORS[gpu], linewidth=2)
    ax.set_xlabel("Matrix size (N, for N×N×N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Naive Kernel: Non-Monotonic Scaling with Matrix Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/03_naive_nonmonotonic.png")
    plt.close()
    print("Saved 03_naive_nonmonotonic.png")


# --- Plot 4: Kernel 5 tile-size/SM cliff ---
def plot_cliff():
    sizes = [1024, 1536, 2048, 4096]
    k4 = [16602.47, 15607.47, 17160.10, 18917.99]
    k5 = [15552.82, 19048.18, 30323.81, 31987.02]
    blocks = [64, 144, 256, 1024]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, k4, marker='o', label="Kernel 4 (1D blocktiled)", color="#457B9D")
    ax1.plot(sizes, k5, marker='s', label="Kernel 5 (2D blocktiled)", color="#E63946")
    ax1.set_xlabel("Matrix size (N, for N×N×N)")
    ax1.set_ylabel("GFLOPS")
    ax1.set_title("Kernel 5 Performance Cliff: Tile Size vs. SM Occupancy (L40S)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(sizes, blocks, marker='^', linestyle='--', color='gray', alpha=0.6, label="k5 block count")
    ax2.axhline(y=142, color='gray', linestyle=':', alpha=0.8)
    ax2.text(sizes[0], 142, ' 142 SMs (L40S)', va='bottom', fontsize=8, color='gray')
    ax2.set_ylabel("Grid block count (kernel 5)", color='gray')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig("plots/04_tile_size_cliff.png")
    plt.close()
    print("Saved 04_tile_size_cliff.png")


# --- Plot 5: Autotuning sweep results distribution ---
def plot_autotuning():
    with open("profiling/autotune_results.json") as f:
        data = json.load(f)
    correct = [r for r in data if r["status"] == "OK"]
    gflops = sorted([r["gflops"] for r in correct], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(gflops)), gflops, color="#2E86AB", width=0.8)
    ax.set_xlabel("Configuration rank (sorted, best to worst)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"Autotuning Sweep: {len(gflops)} Correctness-Verified Configurations (L40S, 2048³)")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/05_autotuning_sweep.png")
    plt.close()
    print("Saved 05_autotuning_sweep.png")


# --- Plot 6: cuBLAS comparison ---
def plot_cublas_comparison(rows):
    gpus = ["L40S", "A100", "RTX6000Ada"]
    custom_best = []
    cublas = []
    for gpu in gpus:
        w = [r for r in rows if r["kernel"] == "Warptiled" and r["gpu"] == gpu]
        c = [r for r in rows if r["kernel"] == "cuBLAS" and r["gpu"] == gpu]
        custom_best.append(w[0]["gflops"] if w else 0)
        cublas.append(c[0]["gflops"] if c else 0)

    x = range(len(gpus))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], custom_best, width, label="Our best kernel (warptiled)", color="#2E86AB")
    ax.bar([i + width/2 for i in x], cublas, width, label="cuBLAS (torch.matmul)", color="#E63946")
    ax.set_xticks(list(x))
    ax.set_xticklabels(gpus)
    ax.set_ylabel("GFLOPS")
    ax.set_title("Our Best Kernel vs. cuBLAS (2048³)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (cb, cu) in enumerate(zip(custom_best, cublas)):
        pct = cb / cu * 100 if cu else 0
        ax.text(i, max(cb, cu) + 1000, f"{pct:.1f}%", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/06_cublas_comparison.png")
    plt.close()
    print("Saved 06_cublas_comparison.png")


if __name__ == "__main__":
    rows = load_consolidated()
    plot_kernel_progression(rows)
    plot_pct_peak_progression(rows)
    plot_naive_scaling(rows)
    plot_cliff()
    plot_autotuning()
    plot_cublas_comparison(rows)
    print("\nAll plots generated in plots/")


# --- Plot 7: Non-square dimension results across architectures ---
def plot_nonsquare_results():
    shapes = ["3000×1500\n×2048", "8192×256\n×1024", "2048×2048\n×2049", "137×263\n×401"]
    blocks = [288, 128, 256, 6]
    l40s = [27515.57, 34083.58, 35932.00, 623.98]
    a100 = [10613.21, 7083.78, 9881.91, 218.59]
    rtx6000 = [30582.82, 38043.57, 42540.81, 625.24]

    x = range(len(shapes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], l40s, width, label="L40S", color=GPU_COLORS["L40S"])
    ax.bar(x, a100, width, label="A100", color=GPU_COLORS["A100"])
    ax.bar([i + width for i in x], rtx6000, width, label="RTX 6000 Ada", color=GPU_COLORS["RTX6000Ada"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}\n({b} blocks)" for s, b in zip(shapes, blocks)], fontsize=8)
    ax.set_ylabel("GFLOPS")
    ax.set_yscale('log')
    ax.set_title("Non-Square / Non-Power-of-Two Dimension Sweep (by architecture)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("plots/07_nonsquare_dimensions.png")
    plt.close()
    print("Saved 07_nonsquare_dimensions.png")


# --- Plot 8: Real ncu hardware metrics (T4) ---
def plot_hardware_metrics():
    kernels = ["Naive", "Coalesced", "SharedMem", "1DBlocktiled", "Vectorized"]
    sectors_per_req = [16.51, 2.50, 4.00, 4.00, 16.81]
    l1_hit = [99.23, 94.93, 1.52, 2.94, 15.32]
    l2_hit = [88.37, 85.42, 74.56, 74.75, 83.59]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(kernels, sectors_per_req, color="#457B9D")
    axes[0].set_ylabel("Sectors per request")
    axes[0].set_title("Global Load Coalescing Efficiency\n(lower = better)")
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].plot(kernels, l1_hit, marker='o', color="#E63946", label="L1 hit rate")
    axes[1].plot(kernels, l2_hit, marker='s', color="#2E86AB", label="L2 hit rate")
    axes[1].set_ylabel("Hit rate (%)")
    axes[1].set_title("Cache Hit Rates")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(True, alpha=0.3)

    bank_conflicts = [0, 0, 0, 0, 4194304]
    axes[2].bar(kernels, bank_conflicts, color="#06A77D")
    axes[2].set_ylabel("Shared-mem bank conflicts (LD)")
    axes[2].set_title("Bank Conflicts by Kernel")
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle("Real Hardware Counter Data (Nsight Compute, Tesla T4)", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/08_hardware_metrics_ncu.png", bbox_inches='tight')
    plt.close()
    print("Saved 08_hardware_metrics_ncu.png")


plot_nonsquare_results()
plot_hardware_metrics()

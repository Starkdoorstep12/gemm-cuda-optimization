"""
Consolidate all kernel-progression results across architectures into one CSV
for plotting.
"""
import csv

# Manually consolidated from README tables (single source of truth going forward)
rows = [
    # kernel, size, gpu, gflops, pct_peak
    ("Naive", 1024, "L40S", 618.16, 0.7),
    ("Naive", 4092, "L40S", 2056.89, 2.2),
    ("Naive", 8192, "L40S", 687.48, 0.8),
    ("Naive", 1024, "A100", 214.38, 1.1),
    ("Naive", 4092, "A100", 874.90, 4.5),
    ("Naive", 8192, "A100", 293.16, 1.5),
    ("Naive", 1024, "RTX6000Ada", 613.28, 0.7),
    ("Naive", 4092, "RTX6000Ada", 2107.06, 2.3),
    ("Naive", 8192, "RTX6000Ada", 740.68, 0.8),

    ("Coalesced", 1024, "L40S", 5028.50, 5.5),
    ("Coalesced", 4092, "L40S", 5249.52, 5.7),
    ("Coalesced", 1024, "A100", 3563.55, 18.3),
    ("Coalesced", 4092, "A100", 3593.40, 18.4),
    ("Coalesced", 1024, "RTX6000Ada", 5363.99, 5.9),
    ("Coalesced", 4092, "RTX6000Ada", 5084.32, 5.6),

    ("SharedMem", 1024, "L40S", 6500.21, 7.1),
    ("SharedMem", 4096, "L40S", 7220.58, 7.9),
    ("SharedMem", 1024, "A100", 5196.12, 26.6),
    ("SharedMem", 4096, "A100", 5446.22, 27.9),
    ("SharedMem", 1024, "RTX6000Ada", 6995.17, 7.7),
    ("SharedMem", 4096, "RTX6000Ada", 7371.63, 8.1),

    ("1DBlocktiled", 1024, "L40S", 16602.47, 18.1),
    ("1DBlocktiled", 1536, "L40S", 15607.47, 17.0),
    ("1DBlocktiled", 2048, "L40S", 17160.10, 18.7),
    ("1DBlocktiled", 4096, "L40S", 18917.99, 20.7),
    ("1DBlocktiled", 1024, "A100", 7925.75, 40.7),
    ("1DBlocktiled", 1536, "A100", 8727.36, 44.8),
    ("1DBlocktiled", 2048, "A100", 7718.63, 39.6),
    ("1DBlocktiled", 4096, "A100", 8918.65, 45.7),
    ("1DBlocktiled", 1024, "RTX6000Ada", 19504.99, 21.4),
    ("1DBlocktiled", 1536, "RTX6000Ada", 18574.09, 20.4),
    ("1DBlocktiled", 2048, "RTX6000Ada", 20901.96, 22.9),
    ("1DBlocktiled", 4096, "RTX6000Ada", 18597.67, 20.4),

    ("2DBlocktiled", 1024, "L40S", 15552.82, 17.0),
    ("2DBlocktiled", 1536, "L40S", 19048.18, 20.8),
    ("2DBlocktiled", 2048, "L40S", 30323.81, 33.1),
    ("2DBlocktiled", 4096, "L40S", 31987.02, 34.9),
    ("2DBlocktiled", 1024, "A100", 7642.68, 39.2),
    ("2DBlocktiled", 1536, "A100", 9478.89, 48.6),
    ("2DBlocktiled", 2048, "A100", 11246.29, 57.7),
    ("2DBlocktiled", 4096, "A100", 11706.84, 60.0),
    ("2DBlocktiled", 1024, "RTX6000Ada", 15385.59, 16.9),
    ("2DBlocktiled", 1536, "RTX6000Ada", 20001.45, 22.0),
    ("2DBlocktiled", 2048, "RTX6000Ada", 34566.14, 37.9),
    ("2DBlocktiled", 4096, "RTX6000Ada", 30527.75, 33.5),

    ("BestAutotuned", 2048, "L40S", 35965.95, 39.3),
    ("BestAutotuned", 2048, "A100", 11274.25, 57.8),
    ("BestAutotuned", 2048, "RTX6000Ada", 42419.93, 46.6),

    ("Warptiled", 2048, "L40S", 37860.87, 41.3),
    ("Warptiled", 2048, "A100", 11545.02, 59.2),
    ("Warptiled", 2048, "RTX6000Ada", 45069.00, 49.5),

    ("cuBLAS", 2048, "L40S", 44340.41, 48.4),
    ("cuBLAS", 2048, "A100", 13734.28, 70.4),
    ("cuBLAS", 2048, "RTX6000Ada", 48572.70, 53.3),
]

with open("profiling/consolidated_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["kernel", "size", "gpu", "gflops", "pct_peak"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to profiling/consolidated_results.csv")

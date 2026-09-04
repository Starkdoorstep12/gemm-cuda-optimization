import torch
import numpy as np
import sys

M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed = 42

torch.manual_seed(seed)
A = torch.rand(M, K, dtype=torch.float32)
B = torch.rand(K, N, dtype=torch.float32)
C_ref = (A @ B).numpy()

A.numpy().tofile(f"profiling/nonsquare_test/A_{M}_{N}_{K}.bin")
B.numpy().tofile(f"profiling/nonsquare_test/B_{M}_{N}_{K}.bin")
C_ref.tofile(f"profiling/nonsquare_test/C_ref_{M}_{N}_{K}.bin")

print(f"Generated A({M}x{K}), B({K}x{N}), reference C({M}x{N})")
print(f"C_ref[0][0] = {C_ref[0][0]}")
print(f"C_ref[-1][-1] = {C_ref[-1][-1]}")

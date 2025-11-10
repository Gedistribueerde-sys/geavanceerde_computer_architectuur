import matplotlib.pyplot as plt
import numpy as np

# Matrix sizes
N = [8, 16, 32, 64, 96, 192, 384, 768]

# Data for BLOCK_SIZE = 16 (averaged where possible from provided outputs)
global_16 = [(0.0044 + 0.0044)/2, (0.0048 + 0.0045)/2, (0.0050 + 0.0053)/2, (0.0065 + 0.0065)/2, (0.0095 + 0.0092)/2, (0.0295 + 0.0298)/2, (0.1904 + 0.1904)/2, (1.3829 + 1.3728)/2]
shared_16 = [np.nan, (0.0046 + 0.0047)/2, (0.0049 + 0.0050)/2, (0.0059 + 0.0060)/2, (0.0084 + 0.0089)/2, (0.0264 + 0.0294)/2, (0.1441 + 0.1437)/2, (0.8157 + 0.8803)/2]
load_16 = [np.nan, (0.0039 + 0.0080)/2, (0.0036 + 0.0038)/2, (0.0042 + 0.0039)/2, (0.0039 + 0.0044)/2, (0.0062 + 0.0057)/2, (0.0105 + 0.0104)/2, (0.0457 + 0.0465)/2]
constant_16 = [(0.0047 + 0.0046)/2, (0.0074 + 0.0074)/2, (0.0133 + 0.0108)/2, (0.0228 + 0.0257)/2, (0.0399 + 0.0402)/2, np.nan, np.nan, np.nan]

# Data for BLOCK_SIZE = 4 (averaged from third and fourth outputs)
global_4 = [(0.0035 + 0.0042)/2, (0.0033 + 0.0043)/2, (0.0040 + 0.0052)/2, (0.0056 + 0.0072)/2, (0.0108 + 0.0136)/2, (0.0445 + 0.0623)/2, (0.3166 + 0.4358)/2, (2.6659 + 3.0103)/2]
shared_4 = [(0.0035 + 0.0044)/2, (0.0036 + 0.0049)/2, (0.0045 + 0.0057)/2, (0.0069 + 0.0080)/2, (0.0127 + 0.0176)/2, (0.0590 + 0.0798)/2, (0.4025 + 0.5404)/2, (3.4626 + 3.4681)/2]
load_4 = [(0.0030 + 0.0037)/2, (0.0035 + 0.0038)/2, (0.0032 + 0.0042)/2, (0.0035 + 0.0043)/2, (0.0049 + 0.0060)/2, (0.0206 + 0.0234)/2, (0.1068 + 0.1457)/2, (0.8425 + 0.8385)/2]
constant_4 = [(0.0035 + 0.0044)/2, (0.0038 + 0.0048)/2, (0.0046 + 0.0060)/2, (0.0078 + 0.0106)/2, (0.0219 + 0.0270)/2, np.nan, np.nan, np.nan]

# Average memory transfer times (H2D A, H2D B, H2Const) - already averaged in previous, keep same
h2d_a = [
    (0.0073 + 0.0058 + 0.0053 + 0.0058) / 4,
    (0.0060 + 0.0055 + 0.0051 + 0.0059) / 4,
    (0.0074 + 0.0064 + 0.0059 + 0.0062) / 4,
    (0.0127 + 0.0087 + 0.0105 + 0.0084) / 4,
    (0.0183 + 0.0163 + 0.0146 + 0.0121) / 4,
    (0.0306 + 0.1554 + 0.0274 + 0.0248) / 4,  # Note: 0.1554 may be an outlier, but keeping
    (0.3052 + 0.1688 + 0.1298 + 0.1600) / 4,
    (0.8889 + 0.4909 + 0.4406 + 0.5059) / 4
]

h2d_b = [
    (0.0053 + 0.0053 + 0.0051 + 0.0058) / 4,
    (0.0059 + 0.0054 + 0.0052 + 0.0057) / 4,
    (0.0063 + 0.0063 + 0.0058 + 0.0062) / 4,
    (0.0089 + 0.0091 + 0.0102 + 0.0089) / 4,
    (0.0142 + 0.0152 + 0.0158 + 0.0126) / 4,
    (0.0242 + 0.0326 + 0.0244 + 0.0223) / 4,
    (0.1244 + 0.1475 + 0.1536 + 0.1069) / 4,
    (0.7487 + 0.5873 + 0.7230 + 0.5777) / 4
]

h2const = [
    (0.0030 + 0.0028 + 0.0026 + 0.0026) / 4,
    (0.0020 + 0.0018 + 0.0018 + 0.0017) / 4,
    (0.0017 + 0.0017 + 0.0017 + 0.0017) / 4,
    (0.0015 + 0.0018 + 0.0017 + 0.0017) / 4,
    (0.0021 + 0.0021 + 0.0020 + 0.0020) / 4,
    np.nan,
    np.nan,
    np.nan
]

# Enhanced Plot 1: Kernel times for BLOCK_SIZE=16
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(N, global_16, label='Global', marker='o', linewidth=2, markersize=5)
ax1.plot(N, shared_16, label='Shared', marker='s', linewidth=2, markersize=5)
ax1.plot(N, constant_16, label='Constant', marker='^', linewidth=2, markersize=5)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Matrix Size N (log scale)', fontsize=12)
ax1.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
ax1.set_title('Kernel Execution Times (BLOCK_SIZE=16)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('kernel_times_block16.png', dpi=300)
plt.close(fig1)

# Enhanced Plot 2: Kernel times for BLOCK_SIZE=4
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(N, global_4, label='Global', marker='o', linewidth=2, markersize=5)
ax2.plot(N, shared_4, label='Shared', marker='s', linewidth=2, markersize=5)
ax2.plot(N, constant_4, label='Constant', marker='^', linewidth=2, markersize=5)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Matrix Size N (log scale)', fontsize=12)
ax2.set_ylabel('Execution Time (ms, log scale)', fontsize=12)
ax2.set_title('Kernel Execution Times (BLOCK_SIZE=4)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('kernel_times_block4.png', dpi=300)
plt.close(fig2)

# Plot 3: General Memory transfer times (as before)
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(N, h2d_a, label='Host to Device (A)', marker='o', linewidth=2, markersize=5)
ax3.plot(N, h2d_b, label='Host to Device (B)', marker='s', linewidth=2, markersize=5)
ax3.plot(N, h2const, label='Host to Constant (B)', marker='^', linewidth=2, markersize=5)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Matrix Size N (log scale)', fontsize=12)
ax3.set_ylabel('Transfer Time (ms, log scale)', fontsize=12)
ax3.set_title('Average Memory Transfer Times (CPU to GPU)', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('memory_transfer_times.png', dpi=300)
plt.close(fig3)

# New Plot 4: Specific Memory Copies Comparison
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(N, h2d_b, label='Normal Memory Copy (Host to Device B)', marker='o', linewidth=2, markersize=5)
ax4.plot(N, h2const, label='Constant Memory Copy (Host to Constant B)', marker='s', linewidth=2, markersize=5)
ax4.plot(N, load_16, label='Shared Loading (Global to Shared, BLOCK=16) for matrix A and B', marker='^', linewidth=2, markersize=5)
ax4.plot(N, load_4, label='Shared Loading (Global to Shared, BLOCK=4) for matrix A and B', marker='d', linewidth=2, markersize=5)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Matrix Size N (log scale)', fontsize=12)
ax4.set_ylabel('Time (ms, log scale)', fontsize=12)
ax4.set_title('Memory Operations Times Comparison', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('memory_operations_comparison.png', dpi=300)
plt.close(fig4)

print("Plots saved as 'kernel_times_block16.png', 'kernel_times_block4.png', 'memory_transfer_times.png', and 'memory_operations_comparison.png'")
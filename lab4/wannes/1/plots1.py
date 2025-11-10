import matplotlib.pyplot as plt
import numpy as np

# Data from three runs
threads = [64, 128, 256, 512, 1024]

# Run data [run1, run2, run3]
data_1x_uncoal = [
    [0.044061, 0.037926, 0.035088, 0.038792, 0.044903],
    [0.041486, 0.034002, 0.032753, 0.035047, 0.037321],
    [0.042829, 0.033909, 0.033493, 0.037581, 0.042971]
]
data_1x_coal = [
    [0.037792, 0.026644, 0.024703, 0.021533, 0.029222],
    [0.036630, 0.020789, 0.017691, 0.018050, 0.022047],
    [0.032487, 0.020774, 0.017166, 0.023623, 0.028167]
]
data_2x_uncoal = [
    [0.066180, 0.060909, 0.060678, 0.055043, 0.074340],
    [0.060245, 0.054530, 0.054811, 0.055948, 0.061837],
    [0.066871, 0.061583, 0.055463, 0.055065, 0.064776]
]
data_2x_coal = [
    [0.059845, 0.037655, 0.039005, 0.041383, 0.054468],
    [0.051393, 0.036354, 0.036421, 0.036925, 0.051087],
    [0.057993, 0.036134, 0.036052, 0.037106, 0.056949]
]
data_4x_uncoal = [
    [0.119852, 0.103291, 0.108636, 0.103992, 0.116871],
    [0.110457, 0.103187, 0.103321, 0.103945, 0.117159],
    [0.115600, 0.112205, 0.102577, 0.114141, 0.116859]
]
data_4x_coal = [
    [0.100675, 0.063161, 0.064429, 0.063756, 0.092035],
    [0.093671, 0.062698, 0.062124, 0.063576, 0.091885],
    [0.095282, 0.062324, 0.065067, 0.067503, 0.091802]
]

# Calculate averages and speedups
avg_1x_uncoal, avg_1x_coal = np.mean(data_1x_uncoal, axis=0), np.mean(data_1x_coal, axis=0)
avg_2x_uncoal, avg_2x_coal = np.mean(data_2x_uncoal, axis=0), np.mean(data_2x_coal, axis=0)
avg_4x_uncoal, avg_4x_coal = np.mean(data_4x_uncoal, axis=0), np.mean(data_4x_coal, axis=0)

speedup_1x = avg_1x_uncoal / avg_1x_coal
speedup_2x = avg_2x_uncoal / avg_2x_coal
speedup_4x = avg_4x_uncoal / avg_4x_coal

# Create figure with 3 essential plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Speedup comparison (most important for report)
ax1.plot(threads, speedup_1x, 'o-', label='1x (3.52 MB)', linewidth=2.5, markersize=9, color='#2ecc71')
ax1.plot(threads, speedup_2x, 's-', label='2x (7.03 MB)', linewidth=2.5, markersize=9, color='#f39c12')
ax1.plot(threads, speedup_4x, '^-', label='4x (14.06 MB)', linewidth=2.5, markersize=9, color='#9b59b6')
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.set_xlabel('Threads per Block', fontsize=12)
ax1.set_ylabel('Speedup Factor', fontsize=12)
ax1.set_title('Coalesced Memory Speedup', fontsize=13, fontweight='bold')
ax1.set_xticks(threads)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Execution time comparison
ax2.plot(threads, avg_1x_uncoal, 'o--', label='Uncoalesced 1x', linewidth=2, markersize=7, color='#2ecc71', alpha=0.7)
ax2.plot(threads, avg_1x_coal, 'o-', label='Coalesced 1x', linewidth=2.5, markersize=7, color='#2ecc71')
ax2.plot(threads, avg_2x_uncoal, 's--', label='Uncoalesced 2x', linewidth=2, markersize=7, color='#f39c12', alpha=0.7)
ax2.plot(threads, avg_2x_coal, 's-', label='Coalesced 2x', linewidth=2.5, markersize=7, color='#f39c12')
ax2.plot(threads, avg_4x_uncoal, '^--', label='Uncoalesced 4x', linewidth=2, markersize=7, color='#9b59b6', alpha=0.7)
ax2.plot(threads, avg_4x_coal, '^-', label='Coalesced 4x', linewidth=2.5, markersize=7, color='#9b59b6')
ax2.set_xlabel('Threads per Block', fontsize=12)
ax2.set_ylabel('Execution Time (ms)', fontsize=12)
ax2.set_title('Execution Time vs Thread Configuration', fontsize=13, fontweight='bold')
ax2.set_xticks(threads)
ax2.legend(fontsize=9, loc='best', ncol=2)
ax2.grid(True, alpha=0.3)

# Plot 3: Performance gap (uncoalesced - coalesced)
perf_gap_1x = avg_1x_uncoal - avg_1x_coal
perf_gap_2x = avg_2x_uncoal - avg_2x_coal
perf_gap_4x = avg_4x_uncoal - avg_4x_coal
ax3.plot(threads, perf_gap_1x, 'o-', label='1x (3.52 MB)', linewidth=2.5, markersize=9, color='#2ecc71')
ax3.plot(threads, perf_gap_2x, 's-', label='2x (7.03 MB)', linewidth=2.5, markersize=9, color='#f39c12')
ax3.plot(threads, perf_gap_4x, '^-', label='4x (14.06 MB)', linewidth=2.5, markersize=9, color='#9b59b6')
ax3.set_xlabel('Threads per Block', fontsize=12)
ax3.set_ylabel('Time Difference (ms)', fontsize=12)
ax3.set_title('Performance Gap (Uncoalesced - Coalesced)', fontsize=13, fontweight='bold')
ax3.set_xticks(threads)
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3)

plt.suptitle('CUDA Memory Coalescing Performance Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('memory_coalescing_report.png', dpi=300, bbox_inches='tight')
plt.show()
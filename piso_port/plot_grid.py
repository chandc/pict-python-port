import numpy as np
import matplotlib.pyplot as plt
import os

from phase1_grid_metrics import analytical_wavy_grid_mms

# Generate a 20x20x20 grid
n = 20
x, y, z, _, _, _, _, _ = analytical_wavy_grid_mms(n, n, n, Ax=0.1, Ay=0.1, Az=0.1)

# Create a 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Constant k cut (X-Y plane) ---
k_idx = n // 2
x_k = np.array(x[:, :, k_idx], dtype=float)
y_k = np.array(y[:, :, k_idx], dtype=float)
axes[0].set_title(f"Constant k cut (k={k_idx})")
for i in range(n): axes[0].plot(x_k[i, :], y_k[i, :], 'k-', alpha=0.5)
for j in range(n): axes[0].plot(x_k[:, j], y_k[:, j], 'b-', alpha=0.5)
axes[0].set_xlabel("X"); axes[0].set_ylabel("Y"); axes[0].axis('equal')

# --- Constant j cut (X-Z plane) ---
j_idx = n // 2
x_j = np.array(x[:, j_idx, :], dtype=float)
z_j = np.array(z[:, j_idx, :], dtype=float)
axes[1].set_title(f"Constant j cut (j={j_idx})")
for i in range(n): axes[1].plot(x_j[i, :], z_j[i, :], 'k-', alpha=0.5)
for k in range(n): axes[1].plot(x_j[:, k], z_j[:, k], 'r-', alpha=0.5)
axes[1].set_xlabel("X"); axes[1].set_ylabel("Z"); axes[1].axis('equal')

# --- Constant i cut (Y-Z plane) ---
i_idx = n // 2
y_i = np.array(y[i_idx, :, :], dtype=float)
z_i = np.array(z[i_idx, :, :], dtype=float)
axes[2].set_title(f"Constant i cut (i={i_idx})")
for j in range(n): axes[2].plot(y_i[j, :], z_i[j, :], 'b-', alpha=0.5)
for k in range(n): axes[2].plot(y_i[:, k], z_i[:, k], 'r-', alpha=0.5)
axes[2].set_xlabel("Y"); axes[2].set_ylabel("Z"); axes[2].axis('equal')

# Save to the artifacts directory
out_path = '/Users/danielchan/.gemini/antigravity-ide/brain/703d4ece-8e8e-4d5d-ab81-a1422e38161f/wavy_grid_3cuts.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {out_path}")

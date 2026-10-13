import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
length = 10000.0        # Total depth of column [m] (10 km)
nx = 1001               # More nodes this time for more accurate sill sizing
dx = length / (nx - 1)  # Grid spacing [m]
depths = np.linspace(0.0, length, nx)

# Time setup
seconds_per_year = 31536000.0
dt = 0.5 * seconds_per_year # 0.5 year time step for stability & precision
total_years = 100000
total_steps = int(total_years / 0.5)

# Diffusivity
kappa = 1.0e-6          # Thermal diffusivity [m^2/s]
D = np.full(nx, kappa)

r = (D * dt) / (dx**2)
assert np.max(r) <= 0.5, f"Unstable! Max r = {np.max(r):.4f}"

# Initial Conditions + sill setup
T_surface = 15.0
T_base = 300.0
T = np.zeros(nx)
for i in range(nx):
    T[i] = T_surface + (T_base - T_surface) * (depths[i] / length)


# Sill 1: 250m thick at 1 km depth (875m to 1125m)
sill1_top = 875.0
sill1_bottom = 1125.0

# Sill 2: 100m thick at 7 km depth (6950m to 7050m)
sill2_top = 6950.0
sill2_bottom = 7050.0

T_magma = 1100.0        # Magma temperature [°C]

for i, z in enumerate(depths):
    if (sill1_top <= z <= sill1_bottom) or (sill2_top <= z <= sill2_bottom):
        T[i] = T_magma

# Save initial profile 
history = [(0.0, T.copy())]
snapshot_years = [0.0, 1000.0, 5000.0, 20000.0, 100000.0]
snapshot_steps = [int(y / 0.5) for y in snapshot_years]

# Matrix
s = (kappa * dt) / (dx**2)
ab = np.zeros((3, nx))
ab[0, 1:] = -s           # Upper diagonal
ab[1, :] = 1.0 + 2.0*s   # Main diagonal
ab[2, :-1] = -s          # Lower diagonal


ab[1, 0] = 1.0
ab[0, 1] = 0.0
ab[1, -1] = 1.0
ab[2, -2] = 0.0

# Simulation loop
print("Running 10km multi-sill cooling simulation...")
for step in range(1, total_steps + 1):
    b = T.copy()
    b[0] = T_surface
    b[-1] = T_base

    # Solve math
    T = solve_banded((1, 1), ab, b)

    # Record snapshots at specified intervals
    if step in snapshot_steps:
        current_years = step * 0.5
        history.append((current_years, T.copy()))

# Ensure final step is captured 
if total_steps not in snapshot_steps:
    history.append((total_years, T.copy()))

# Graphing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1.5, 1]})
fig.suptitle("10 km Column Thermal Diffusion: Multiple Sills Cooling Model", 
             fontsize=16, fontweight='bold', y=0.95)

# Graph1
for years, T_profile in history:
    if years == 0:
        label_str = 't = 0 yr (Initial Sills)'
    elif years < 100000:
        label_str = f't = {years/1000.0:.1f} kyr'
    else:
        label_str = f't = {years/1000.0:.0f} kyr'
    ax1.plot(T_profile, depths, lw=2.5, label=label_str)

ax1.set_xlabel("Temperature (°C)", color="red", fontweight="bold")
ax1.set_ylabel("Depth (m)", fontweight="bold")
ax1.tick_params(axis='x', colors="red")
ax1.invert_yaxis()
ax1.grid(True, linestyle=":", alpha=0.5)

# Highlight oil + gas window
ax1.axvspan(60, 120, color='green', alpha=0.2, label='Oil Window (60-120°C)')
ax1.axvspan(120, 200, color='orange', alpha=0.2, label='Gas Window (120-200°C)')

# Top x-axis thermal diffusivity
ax_top = ax1.twiny()
ax_top.plot(np.full(nx, kappa * 1e6), depths, "k--", lw=2, alpha=0.7, label="Thermal Diffusivity")
ax_top.set_xlabel("Thermal Diffusivity (x10⁻⁶ m²/s)", color="black", fontweight="bold")
ax_top.tick_params(axis='x', colors="black")


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_top, labels_top = ax_top.get_legend_handles_labels()
ax1.legend(lines_1 + lines_top, labels_1 + labels_top, loc="lower right", fontsize=9)

# Graph2
T_2D = np.tile(T, (10, 1)).T  
img = ax2.imshow(T_2D, aspect='auto', cmap='jet', extent=[0, 1, length, 0])

# Mark the sill boundaries on the heatmap
ax2.axhline(sill1_top, color='white', linestyle='--', lw=1.5)
ax2.axhline(sill1_bottom, color='white', linestyle='--', lw=1.5)
ax2.axhline(sill2_top, color='white', linestyle='--', lw=1.5)
ax2.axhline(sill2_bottom, color='white', linestyle='--', lw=1.5)

ax2.text(0.5, 1000, '250m Sill (1 km)', color='white', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(0.5, 7000, '100m Sill (7 km)', color='white', ha='center', va='center', fontsize=10, fontweight='bold')

ax2.set_title("2D Thermal Cross-Section (Final State)", fontweight="bold")
ax2.set_xticks([]) 
cbar = fig.colorbar(img, ax=ax2, pad=0.05)
cbar.set_label("Temperature (°C)", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
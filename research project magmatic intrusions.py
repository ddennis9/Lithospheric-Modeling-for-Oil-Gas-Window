import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
length = 10000.0        # Total depth of column [m] (10 km)
nx = 501                # Number of depth nodes 
dx = length / (nx - 1)  # Grid spacing [m]
depths = np.linspace(0.0, length, nx)

# Time setup
seconds_per_year = 31536000.0
dt = 1.0 * seconds_per_year # 1 year time step 
total_years = 100000
total_steps = int(total_years / 1.0)

# Diffusivity
kappa = 1.0e-6          # Thermal diffusivity [m^2/s]
D = np.full(nx, kappa)

r = (D * dt) / (dx**2)
assert np.max(r) <= 0.5, f"Unstable! Max r = {np.max(r):.4f}"

# Initial Conditions
T_surface = 0.0
T_base = 300.0
T = np.zeros(nx)
for i in range(nx):
    T[i] = T_surface + (T_base - T_surface) * (depths[i] / length)

# Magma
sill_top = 4750.0
sill_bottom = 5250.0
T_magma = 1100.0        # Magma temperature [°C]

for i, z in enumerate(depths):
    if sill_top <= z <= sill_bottom:
        T[i] = T_magma

# Saving initial profile
history = [(0.0, T.copy())]
snapshot_years = [0.0, 2000.0, 10000.0, 30000.0, 100000.0]
snapshot_steps = [int(y / 1.0) for y in snapshot_years]

# Setup calculations
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
print("Running 10km sill cooling simulation...")
for step in range(1, total_steps + 1):
    b = T.copy()
    b[0] = T_surface
    b[-1] = T_base

    # Solve math
    T = solve_banded((1, 1), ab, b)

    # Record snapshots for graph
    if step in snapshot_steps:
        current_years = step * 1.0
        history.append((current_years, T.copy()))

# Ensure final step is captured 
if total_steps not in snapshot_steps:
    history.append((total_years, T.copy()))

# Graphing stuff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1.5, 1]})
fig.suptitle("10 km Column Thermal Diffusion: 500m Sill Cooling at 5 km", 
             fontsize=16, fontweight='bold', y=0.95)

# Graph1
for years, T_profile in history:
    label_str = 't = 0 yr (Initial Sill)' if years == 0 else f't = {years/1000.0:.1f} kyr'
    ax1.plot(T_profile, depths, lw=2.5, label=label_str)

ax1.set_xlabel("Temperature (°C)", color="red", fontweight="bold")
ax1.set_ylabel("Depth (m)", fontweight="bold")
ax1.tick_params(axis='x', colors="red")
ax1.invert_yaxis()
ax1.grid(True, linestyle=":", alpha=0.5)

# Highlighting oil + gas windowss
ax1.axvspan(60, 120, color='green', alpha=0.2, label='Oil Window (60-120°C)')
ax1.axvspan(120, 200, color='orange', alpha=0.2, label='Gas Window (120-200°C)')

# Top of x-axis showing thermal diffusivity but not really needed for this model as I simplified it so that we don't have to look at different diffusivities
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
ax2.axhline(sill_top, color='white', linestyle='--', lw=1.5)
ax2.axhline(sill_bottom, color='white', linestyle='--', lw=1.5)
ax2.text(0.5, 5000, '500m Sill (5 km)', color='white', ha='center', va='center', fontsize=11, fontweight='bold')

ax2.set_title("2D Thermal Cross-Section (Final State)", fontweight="bold")
ax2.set_xticks([]) 
cbar = fig.colorbar(img, ax=ax2, pad=0.05)
cbar.set_label("Temperature (°C)", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
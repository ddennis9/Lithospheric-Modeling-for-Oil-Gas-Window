import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

#PARAMETERS
length = 8000.0  # (meters) - 8 km lithospheric depth
nx = 401         # [nodes] - 401 nodes for 8km
dx = length / (nx - 1)  # (meters) - 20 meters per node
depths = np.linspace(0.0, length, nx)

#simulating 100,000 years
seconds_per_year = 31536000.0
dt = 2.0 * seconds_per_year  #2 year time step
total_years = 100000
total_steps = int(total_years / 2.0)

#diffusivity values
D = np.zeros(nx)
for i, z in enumerate(depths):
    if z < 2000: D[i] = 1.12e-6   # Upper Sandstones/Sediments
    elif z < 4000: D[i] = 0.84e-6 # Deep Marine Shale
    elif z < 6000: D[i] = 1.20e-6 # Carbonate/Limestone sequence
    else: D[i] = 1.60e-6          #(Granite)

r = (D * dt) / (dx**2)
assert np.max(r) <= 0.5, f"Unstable! Max r = {np.max(r):.4f}"

#MATRIXES
# Surface temp = 15°C. Bottom of 8km crust = 250°C
T = np.full(nx, 15.0)
T[0] = 15.0
T[-1] = 250.0 

M = np.diag(1.0 - 2.0 * r)
M += np.diag(r[1:], k=-1)
M += np.diag(r[:-1], k=1)
M[0, :] = 0.0; M[0, 0] = 1.0
M[-1, :] = 0.0; M[-1, -1] = 1.0

#loop for the simulation
print("Running 8km lithospheric simulation...")
for step in range(1, total_steps + 1):
    T = M @ T

#VISUALIZATIONS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1.5, 1]})
fig.suptitle("Lithospheric Heat Diffusion & Hydrocarbon Windows (8 km)", 
             fontsize=16, fontweight='bold', y=0.95)

#GRAPH 1: Dual-Axis Temperature Profile
# Plot 1: Temperature on bottom X-axis
ax1.plot(T, depths, "r-", lw=3, label="Final Temp (100k yrs)")
ax1.set_xlabel("Temperature (°C)", color="red", fontweight="bold")
ax1.set_ylabel("Depth (m)", fontweight="bold")
ax1.tick_params(axis='x', colors="red")
ax1.invert_yaxis()
ax1.grid(True, linestyle=":", alpha=0.5)

#highlighting hydrocarbon generation windows
ax1.axvspan(60, 120, color='green', alpha=0.2, label='Oil Window (60-120°C)')
ax1.axvspan(120, 200, color='orange', alpha=0.2, label='Gas Window (120-200°C)')

#create a top X-axis for diffusivity
ax_top = ax1.twiny()
ax_top.plot(D * 1e6, depths, "k--", lw=2, alpha=0.7, label="Thermal Diffusivity")
ax_top.set_xlabel("Thermal Diffusivity (x10⁻⁶ m²/s)", color="black", fontweight="bold")
ax_top.tick_params(axis='x', colors="black")

#combine
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_top, labels_top = ax_top.get_legend_handles_labels()
ax1.legend(lines_1 + lines_top, labels_1 + labels_top, loc="lower left")

#graph
T_2D = np.tile(T, (10, 1)).T  

#plotting the heatmap
img = ax2.imshow(T_2D, aspect='auto', cmap='jet', extent=[0, 1, length, 0])

#drawing boundaries for geological layers
ax2.axhline(2000, color='white', linestyle='--', lw=2)
ax2.axhline(4000, color='white', linestyle='--', lw=2)
ax2.axhline(6000, color='white', linestyle='--', lw=2)

#label the layers
ax2.text(0.5, 1000, 'Sandstone', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
ax2.text(0.5, 3000, 'Shale', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
ax2.text(0.5, 5000, 'Limestone', color='white', ha='center', va='center', fontsize=12, fontweight='bold')
ax2.text(0.5, 7000, 'Granite', color='white', ha='center', va='center', fontsize=12, fontweight='bold')

#formatting
ax2.set_title("2D Thermal Cross-Section", fontweight="bold")
ax2.set_xticks([]) 
cbar = fig.colorbar(img, ax=ax2, pad=0.05)
cbar.set_label("Temperature (°C)", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
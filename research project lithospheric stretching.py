import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
a = 120000.0         # Full lithosphere thickness (m)
beta = 2.0           # Stretching factor (e.g., 2.0 = stretched to 200%)
thinned_base = a / beta # Depth of thinned lithosphere base (m)
Tm = 1333.0          # Asthenosphere temperature (°C)
kappa = 1e-6         # Thermal diffusivity (m^2/s)
sec_per_yr = 3.154e7 # Seconds in a year

# Realistic pre streching surface heat flow 
base_geotherm_gradient = 0.030 # 30°C per km (°C/m)

# --- GRID SETUP ---
nz = 401             # num nodes
dz = a / (nz - 1)    # Node spacing (m)
depths = np.linspace(0, a, nz) # Depth array (m)

# Simulation time: 50 Myrs total, stretching occurs at 25 Myr
total_time = 50e6 * sec_per_yr 
nt = 1000
dt = total_time / nt
rift_time_myr = 25.0

# Initial conditions
T = np.minimum(Tm, depths * base_geotherm_gradient)

# Matrixes
s = (kappa * dt) / (dz**2)


A_banded = np.zeros((3, nz))
A_banded[0, 1:] = -s           # Upper diag
A_banded[1, :] = 1 + 2*s       # Main diag
A_banded[2, :-1] = -s          # Lower diag


A_banded[0, 1] = 0.0
A_banded[1, 0] = 1.0


A_banded[1, -1] = 1.0
A_banded[2, -2] = 0.0

# Simulation with the crustal stretching at 25 mil. yrs
print("Running simulation with stretching event at 25 Myr...")
history = []
labels = []

# Save pre-rift state (t = 0 Myr)
history.append(T.copy())
labels.append('t = 0.0 Myr (Pre-Rift)')

for step in range(1, nt + 1):
    current_time_myr = (step * dt) / sec_per_yr / 1e6
    
    # Apply instantaneous McKenzie stretching event at t = 25 Myr
    if abs(current_time_myr - rift_time_myr) < (dt / sec_per_yr / 1e6 / 2):
        print(f"Applying Crustal Stretching (Beta = {beta}) at t = {current_time_myr:.1f} Myr")
        for i in range(nz):
            if depths[i] <= thinned_base:
                # Stretching steepens the gradient in the thinned zone by factor beta
                T[i] = min(Tm, base_geotherm_gradient * beta * depths[i])
            else:
                T[i] = Tm
        history.append(T.copy())
        labels.append(f't = {current_time_myr:.1f} Myr (Rifting Event)')

    b = T.copy() # Right hand side is previous time step
    
    
    b[0] = 0.0
    b[-1] = Tm
    
    # Solve math
    T = solve_banded((1, 1), A_banded, b)
    
    # Save post-rift snapshots to observe how diffusion changes over time
    if abs(current_time_myr - 30.0) < (dt / sec_per_yr / 1e6 / 2) or \
       abs(current_time_myr - 35.0) < (dt / sec_per_yr / 1e6 / 2) or \
       abs(current_time_myr - 40.0) < (dt / sec_per_yr / 1e6 / 2) or \
       abs(current_time_myr - 50.0) < (dt / sec_per_yr / 1e6 / 2):
        history.append(T.copy())
        labels.append(f't = {current_time_myr:.1f} Myr (Post-Rift)')

# Graphing stuff
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1.5, 1]})
fig.suptitle(f"Lithospheric Stretching Event at {rift_time_myr} Myr (Beta = {beta}) - Upper Crust (0-10 km)", 
             fontsize=16, fontweight='bold', y=0.95)

# Graph1
colors = ['blue', 'red', 'darkorange', 'green', 'purple', 'brown']
for idx, (T_profile, label_text) in enumerate(zip(history, labels)):
    c = 'red' if 'Rifting' in label_text else colors[idx % len(colors)]
    lw = 3 if 'Rifting' in label_text or 'Pre-Rift' in label_text else 2
    ax1.plot(T_profile, depths / 1000, color=c, lw=lw, label=label_text)

ax1.set_xlabel("Temperature (°C)", color="red", fontweight="bold")
ax1.set_ylabel("Depth (km)", fontweight="bold")
ax1.tick_params(axis='x', colors="red")
ax1.set_xlim(0, 400)     # Max temperature capped at 400°C
ax1.set_ylim(10, 0)      # Max depth capped at 10 km
ax1.grid(True, linestyle=":", alpha=0.5)

# Highlight oil + gas windows
ax1.axvspan(60, 120, color='green', alpha=0.2, label='Oil Window (60-120°C)')
ax1.axvspan(120, 200, color='orange', alpha=0.2, label='Gas Window (120-200°C)')

# Top of x-axis for thermal diffusivity
D_profile = np.full(nz, kappa)
ax_top = ax1.twiny()
ax_top.plot(D_profile * 1e6, depths / 1000, "k--", lw=2, alpha=0.7, label="Thermal Diffusivity")
ax_top.set_xlabel("Thermal Diffusivity (x10⁻⁶ m²/s)", color="black", fontweight="bold")
ax_top.tick_params(axis='x', colors="black")


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_top, labels_top = ax_top.get_legend_handles_labels()
ax1.legend(lines_1 + lines_top, labels_1 + labels_top, loc="lower left", fontsize=9)

# Graph2
upper_idx = np.where(depths <= 10000.0)[0]
T_upper = T[upper_idx]
depths_upper = depths[upper_idx] / 1000

T_2D = np.tile(T_upper, (10, 1)).T  

img = ax2.imshow(T_2D, aspect='auto', cmap='jet', extent=[0, 1, depths_upper[-1], depths_upper[0]])
ax2.set_ylim(10, 0)

ax2.set_title("2D Thermal Cross-Section (Final State at 50 Myr)", fontweight="bold")
ax2.set_xticks([]) 
ax2.set_ylabel("Depth (km)", fontweight="bold")
cbar = fig.colorbar(img, ax=ax2, pad=0.05)
cbar.set_label("Temperature (°C)", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
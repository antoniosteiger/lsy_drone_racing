import numpy as np
import matplotlib.pyplot as plt
import re

# --- Gate definitions ---
GATE_POSITIONS = np.array([
    [ 0.32048357, -0.4687922,   0.61121744],
    [ 0.91429657, -0.98323965,  1.1039798 ],
    [-0.00207757,  0.9507993,   0.5548653 ],
    [-0.37955815,  0.04865531,  1.1260331 ]
])
GATE_THRESHOLD = 0.15

# --- Load Pos / Goal from file ---
filename = "flight_data_pos2.txt"
positions, goals = [], []

pattern = re.compile(r'\[([^\]]+)\]')

with open(filename, "r") as file:
    for line in file:
        if "Pos:" in line or "Goal:" in line:
            match = pattern.search(line)
            if match:
                vec = np.fromstring(match.group(1), sep=' ')
                if "Pos:" in line:
                    positions.append(vec)
                else:
                    goals.append(vec)

positions = np.array(positions)
goals = np.array(goals)
min_len = min(len(positions), len(goals))
positions = positions[:min_len]
goals = goals[:min_len]

# --- Tracking error ---
errors = []

for i in range(len(goals) - 1):
    g_i = goals[i]
    g_next = goals[i + 1]
    p_i = positions[i]

    v = g_next - g_i           # trajectory direction
    e = p_i - g_i              # error vector
    v_norm_sq = np.dot(v, v)
    
    if v_norm_sq > 1e-8:
        proj = np.dot(e, v) / v_norm_sq * v
        e_orth = e - proj
    else:
        e_orth = e  # fallback if trajectory is stationary

    errors.append(np.linalg.norm(e_orth))

# Pad the last entry to match lengths
errors.append(errors[-1])
errors = np.array(errors)

# --- Gate detection ---
gate_crossings = [[] for _ in GATE_POSITIONS]

for i, goal in enumerate(goals):
    for g_idx, gate in enumerate(GATE_POSITIONS):
        if np.linalg.norm(goal - gate) < GATE_THRESHOLD:
            gate_crossings[g_idx].append(i)

# --- Plot ---
plt.rcParams.update({
    "font.size": 14,              # Base font size
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14
})
plt.figure(figsize=(10, 4))
plt.plot(errors, label="Tracking Error", color='blue')

# --- Convert gate crossings into bands ---
from collections import defaultdict

# Collect matching indices for each gate
gate_match_indices = defaultdict(list)

for i, g_i in enumerate(goals):
    for g_idx, gate in enumerate(GATE_POSITIONS):
        if np.linalg.norm(g_i - gate) < GATE_THRESHOLD:
            gate_match_indices[g_idx].append(i)

# Plot background bands for each gate crossing region
colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff66']  # soft background colors

for g_idx, indices in gate_match_indices.items():
    if not indices:
        continue
    start = min(indices)
    end = max(indices)
    plt.axvspan(start, end, color=colors[g_idx % len(colors)], alpha=0.4, label=f'Gate {g_idx+1}')



plt.xlabel("Time step")
plt.ylabel("Error (meters)")
plt.title("Drone Tracking Error with Gate Crossings")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("tracking_error_plot.pdf", bbox_inches='tight')

plt.show()

# --- Optional: print gate crossing indices ---
for g_idx, crossings in enumerate(gate_crossings):
    print(f"Gate {g_idx+1} crossed at time steps: {crossings}")
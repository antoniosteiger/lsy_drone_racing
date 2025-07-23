import numpy as np
import matplotlib.pyplot as plt

# Load both files (space-separated, variable precision ok)
flight_data = np.loadtxt("flight_data_pos.txt")         # shape: (N, 3)
original_path = np.loadtxt("trajectory.txt")     # shape: (N, 3)

# Ensure they have the same length
min_len = min(len(flight_data), len(original_path))
flight_data = flight_data[:min_len]
original_path = original_path[:min_len]

# Compute tracking error (Euclidean distance per time step)
errors = np.linalg.norm(flight_data - original_path, axis=1)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(errors, label="Tracking Error")
plt.xlabel("Time step")
plt.ylabel("Error (meters)")
plt.title("Drone Tracking Error Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController  # Adjust if needed
import types

# Dummy gate setup (you can use real ones from config if available)
dummy_gates = [np.array([i, 0.5*i, 1.0]) for i in range(1, 5)]

# Dummy obs and config
obs = {"pos": np.array([0.0, 0.0, 0.0]), "vel": np.array([0.0, 0.0, 0.0])}
info = {}
class DummyConfig:
    class env:
        freq = 50
        class track:
            gates = [types.SimpleNamespace(pos=g) for g in dummy_gates]
            obstacles = []

config = DummyConfig()

# Create the controller instance
controller = MinSnapMPCController(obs, info, config)

# Plot trajectory
pos = controller._reference_trajectory["pos"]
t = controller._reference_trajectory["t"]

plt.figure(figsize=(10, 5))
plt.plot(t, pos[:, 0], label="x")
plt.plot(t, pos[:, 1], label="y")
plt.plot(t, pos[:, 2], label="z")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Reference Trajectory from MinSnap")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_plot.png", dpi=300)
print("Plot saved to trajectory_plot.png")

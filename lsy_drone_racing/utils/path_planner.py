import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
import matplotlib.pyplot as plt

class PathPlanner:
    def __init__(self):
        self.num_points = 400
        self.gates_pos = None
        self.obstacles = None
        self.path = None

    def plan(self, gates_pos, gates_rpy = None, obstacles = None, collision_dist=0.15):
        self.gates_pos = gates_pos
        self.obstacles = obstacles
        
        tangents = self.rpy_to_tangents(gates_rpy, 2.0)
        # Generate path
        p = np.arange(len(gates_pos)) # parametrization
        if gates_rpy is not None:
            splines = [CubicHermiteSpline(p, gates_pos[:, i], tangents[:, i]) for i in range(3)]
        else:
            splines = [CubicSpline(p, gates_pos[:, i]) for i in range(3)]
        p_fine = np.linspace(p[0], p[-1], self.num_points)
        path = np.stack([spline(p_fine) for spline in splines], axis=1)
        self.path = path

        if obstacles is None:
            self.path = path
            return path

        # Collision avoidance
        diff = path[:, None, :] - obstacles[None, :, :]  # shape (num_points, num_obstacles, 3)
        dists = np.linalg.norm(diff, axis=2)  # shape (num_points, num_obstacles)
        collision_indices = np.where(dists <= collision_dist)

        if len(collision_indices[0]) == 0:
            # no collision
            self.path = path
            return path

        print("Collision point indices:", collision_indices[0])
        print("Collision obstacle indices:", collision_indices[1])
        return path
    
    def plot(self):
        obstacle_width=0.10
        if self.path is None:
            return
        
        def set_axes_equal(ax):
            '''Set 3D plot axes to equal scale.'''
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            spans = limits[:,1] - limits[:,0]
            centers = np.mean(limits, axis=1)
            radius = 0.5 * max(spans)
            ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
            ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
            ax.set_zlim3d(centers[2] - radius, centers[2] + radius)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*self.path.T)
        ax.scatter(*self.gates_pos.T, color='green')
        if self.obstacles is not None:
            for (x, y, _) in self.obstacles:
                xs = np.array([x - obstacle_width/2, x + obstacle_width/2, x + obstacle_width/2, x - obstacle_width/2, x - obstacle_width/2])
                ys = np.array([y - obstacle_width/2, y - obstacle_width/2, y + obstacle_width/2, y + obstacle_width/2, y - obstacle_width/2])
                zs_bottom = np.zeros_like(xs)
                zs_top = np.ones_like(xs) * 1.4

                # Draw bottom square
                ax.plot(xs, ys, zs_bottom, color='red')

                # Draw top square
                ax.plot(xs, ys, zs_top, color='red')

                # Draw vertical edges
                for i in range(len(xs)-1):
                    ax.plot([xs[i], xs[i]], [ys[i], ys[i]], [0, 1.4], color='red')
        
        set_axes_equal(ax)
        plt.show()

    def rpy_to_tangents(self, rpy_array, magnitude=1.0):
        yaw = rpy_array[:, 2] + np.pi / 2  # Adjust direction to fly into gate
        yaw[0] = yaw[0] - np.pi # first yaw entry is the drone, need to fly downwards
        dx = np.cos(yaw)
        dy = np.sin(yaw)
        dz = np.zeros_like(yaw)
        return magnitude * np.stack([dx, dy, dz], axis=1)


# Testing
# pp = PathPlanner()

# gates_pos = np.array([
#     [1.0, 1.5, 0.07],
#     [0.45, -0.5, 0.56],
#     [1.0, -1.05, 1.11],
#     [0.0, 1.0, 0.56],
#     [-0.5, 0.0, 1.11]
# ])

# gates_rpy = np.array([
#     [0, 0, 0],
#     [0.0, 0.0, 2.35],
#     [0.0, 0.0, -0.78],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 3.14]

# ])

# obstacles = np.array([
#     [1.0, 0.0, 1.4],
#     [0.5, -1.0, 1.4],
#     [0.0, 1.5, 1.4],
#     [-0.5, 0.5, 1.4]
# ])

# pp.plan(gates_pos, gates_rpy, obstacles)
# pp.plot()
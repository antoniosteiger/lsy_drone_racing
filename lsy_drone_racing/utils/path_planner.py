import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class PathPlanner:
    def __init__(self, pos, gates_pos, gates_rpy, obstacles_pos):
        self.POINTS_PER_SECTION = 100
        self.OBSTACLE_SIZE = [0.15, 0.15, 2.0]
        self.GATE_SIZE = [0.57, 0.03, 0.75]
        
        self.initial_pos = pos
        self.gates_pos = np.vstack([pos, gates_pos])
        self.gates_rpy = np.vstack([[0, 0, 0], gates_rpy])
        self.obstacles = self.make_obstacles(self.gates_pos, self.gates_rpy, obstacles_pos, self.GATE_SIZE, self.OBSTACLE_SIZE)
        
        self.collision_points = []
        self.collision_indices = []
        self.collision_obstacles = []

        self.path = self.make_path(self.gates_pos, self.gates_rpy)
        self.collision_points, self.collision_indices, self.collision_obstacles = self.check_collisions(self.path, self.obstacles)
        self.path = self.reroute(self.path, self.collision_indices, self.collision_points, self.collision_obstacles)

    def update(self, gates_pos, gates_rpy, obstacles_pos):
        print("starting path update")
        self.gates_pos = np.vstack([self.initial_pos, gates_pos])
        self.gates_rpy = np.vstack([[0, 0, 0], gates_rpy])
        self.obstacles = self.make_obstacles(self.gates_pos, self.gates_rpy, obstacles_pos, self.GATE_SIZE, self.OBSTACLE_SIZE)
        print("1 initialized variables")
        self.collision_points = []
        self.collision_indices = []
        self.collision_obstacles = []

        # print(self.obstacles)
        self.path = self.make_path(self.gates_pos, self.gates_rpy)
        print("2 made path")
        self.collision_points, self.collision_indices, self.collision_obstacles = self.check_collisions(self.path, self.obstacles)
        print("3 detected collisions")
        self.path = self.reroute(self.path, self.collision_indices, self.collision_points, self.collision_obstacles)
        print("4 rerouted")
        
        print("path update complete")

        return self.path
    
    def make_path(self, gates_pos, gates_rpy):
        tangents = self.rpy_to_tangents(gates_rpy, 2.0)
        # Generate path
        p = np.arange(len(gates_pos)) # parametrization
        if gates_rpy is not None:
            splines = [CubicHermiteSpline(p, gates_pos[:, i], tangents[:, i]) for i in range(3)]
        else:
            splines = [CubicSpline(p, gates_pos[:, i]) for i in range(3)]
        p_fine = np.linspace(p[0], p[-1], self.POINTS_PER_SECTION * (len(gates_pos) -1))
        path = np.stack([spline(p_fine) for spline in splines], axis=1)
        self.path = path

        return path
    
    def check_collisions(self, path, obstacles, margin=0.1, debug=False):
        """
        Check for collisions between a 3D path and rotated cuboid obstacles.
        Returns the first collision point found for each obstacle.
        
        Args:
            path: numpy array of shape (N, 3) - trajectory points
            obstacles: list of obstacle dictionaries, each containing:
                - 'center': [x, y, z] center position
                - 'size': [width, depth, height] dimensions (x, y, z)
                - 'rotation': [roll, pitch, yaw] in radians (optional, defaults to [0,0,0])
            margin: float - safety margin around obstacles (default 0.05)
            debug: bool - print debug information
        
        Returns:
            numpy array of first collision point for each obstacle that has collisions
        """
        collision_indices = []
        collision_points = []
        collision_obstacles = []
        
        for i, obstacle in enumerate(obstacles):
            center = np.array(obstacle['center'])
            size = np.array(obstacle['size'])  # [width, depth, height] = [x, y, z]
            rotation = obstacle.get('rotation', [0, 0, 0])
            
            if debug:
                print(f"\n=== Obstacle {i} ===")
                print(f"Center: {center}")
                print(f"Size: {size}")
                print(f"Rotation: {rotation}")
                print(f"Margin: {margin}")
            
            # Add margin to obstacle size
            expanded_size = size + 2 * margin
            half_size = expanded_size / 2
            
            if debug:
                print(f"Expanded size: {expanded_size}")
                print(f"Half size: {half_size}")
            
            # Transform path points to obstacle's local coordinate system
            relative_points = path - center
            
            if debug:
                print(f"First few relative points: {relative_points[:3]}")
            
            # Apply inverse rotation if obstacle is rotated
            if np.any(rotation):
                rotation_matrix = R.from_euler('XYZ', rotation).as_matrix()
                local_points = relative_points @ rotation_matrix
                if debug:
                    print(f"Applied rotation matrix")
            else:
                local_points = relative_points
                if debug:
                    print(f"No rotation applied")
            
            if debug:
                print(f"First few local points: {local_points[:3]}")
            
            # Check if any points are within expanded obstacle bounds
            # A point is inside if ALL coordinates are within bounds
            within_x = np.abs(local_points[:, 0]) <= half_size[0]
            within_y = np.abs(local_points[:, 1]) <= half_size[1] 
            within_z = np.abs(local_points[:, 2]) <= half_size[2]
            
            within_bounds = within_x & within_y & within_z
            
            if debug:
                print(f"Within X bounds: {within_x[:5]}")
                print(f"Within Y bounds: {within_y[:5]}")
                print(f"Within Z bounds: {within_z[:5]}")
                print(f"Within ALL bounds: {within_bounds[:5]}")
            
            # Find first collision point
            col_i = np.where(within_bounds)[0]
            if len(col_i) > 0:
                # only append first collision points and remove "collisions" with gate center:
                last_index = -2  # initialize to something that's not equal to index - 1
                for i, index in enumerate(col_i):
                    if index != last_index + 1 and not (index % self.POINTS_PER_SECTION >= 90 or index % self.POINTS_PER_SECTION <= 10):
                        collision_indices.append(col_i[i])
                        collision_points.append(path[index])
                        collision_obstacles.append(obstacle)

                        if debug:
                            print(f"COLLISION FOUND at index {collision_indices[-1]}")
                            print(f"Collision point: {path[index]}")
                    last_index = index

            else:
                if debug:
                    print("No collision found")
        
        return np.array(collision_points) if collision_points else np.array([]).reshape(0, 3), collision_indices, collision_obstacles

    def reroute(self, path, collision_indices, collision_points, collision_obstacles):
        BACKTRACK = 30
        DISTANCE = 0.4
        SCALE = 0.3
        p = np.arange(3)
        # print(collision_indices)

        for i, index in enumerate(collision_indices):
            
            
            start_derivative = path[index-BACKTRACK] - path[index-BACKTRACK-1]
            start_derivative = (start_derivative / np.linalg.norm(start_derivative)) * SCALE
            end_derivative = path[index+BACKTRACK+1] - path[index+BACKTRACK]
            end_derivative = (end_derivative / np.linalg.norm(end_derivative)) * SCALE
            
            detour_direction = collision_points[i] - collision_obstacles[i]["center"]
            #detour_direction[2] = 0  # Ignore Z axis
            detour_direction = detour_direction / np.linalg.norm(detour_direction) * SCALE
            perpendicular = np.cross(start_derivative, [0, 0, 1])
            perpendicular = (perpendicular / np.linalg.norm(perpendicular)) * SCALE
            #perpendicular[2] = 0
            detour_point = collision_points[i] + 0.3 * detour_direction + 2.0 * perpendicular * DISTANCE
            
            if index + BACKTRACK + 1 >= len(path):
                return path

            points = np.array([
                path[index-BACKTRACK],
                detour_point,
                path[index+BACKTRACK]
            ])
            
            print("4.1 made points")

            splines = [CubicSpline(p, points[:, i], bc_type=((1, start_derivative[i]), (1, end_derivative[i]))) for i in range(3)]
            p_fine = np.linspace(p[0], p[-1], BACKTRACK*2)
            detour = np.stack([spline(p_fine) for spline in splines], axis=1)

            print("4.2 made splines")
            # print(f"{index + BACKTRACK}; {len(path)}")
            
            path[index-BACKTRACK:index + BACKTRACK] = detour
            print("4.3 Applied reroute")
            print(path)
            
        
        self.collision_indices = []
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
        if len(self.collision_points) > 0:
            ax.scatter(*self.collision_points.T, color='red')

        if self.obstacles is not None:
            for obstacle in self.obstacles:
                center, size, rotation = obstacle['center'], obstacle['size'], obstacle['rotation']
                
                # Extract individual dimensions
                x, y, z = center
                width, depth, height = size  # [x, y, z] dimensions
                
                # Define corners of the cuboid in local coordinates (centered at origin)
                half_width = width / 2
                half_depth = depth / 2
                
                # Bottom face corners (z = -height/2)
                bottom_corners = np.array([
                    [-half_width, -half_depth, -height/2],
                    [half_width, -half_depth, -height/2],
                    [half_width, half_depth, -height/2],
                    [-half_width, half_depth, -height/2],
                    [-half_width, -half_depth, -height/2]  # Close the loop
                ])
                
                # Top face corners (z = +height/2)
                top_corners = np.array([
                    [-half_width, -half_depth, height/2],
                    [half_width, -half_depth, height/2],
                    [half_width, half_depth, height/2],
                    [-half_width, half_depth, height/2],
                    [-half_width, -half_depth, height/2]  # Close the loop
                ])
                
                # Apply rotation if specified
                if np.any(rotation):
                    rot_matrix = R.from_euler('XYZ', rotation).as_matrix()
                    bottom_corners = bottom_corners @ rot_matrix.T
                    top_corners = top_corners @ rot_matrix.T
                
                # Translate to world coordinates
                bottom_corners += center
                top_corners += center
                
                # Extract coordinates for plotting
                xs_bottom = bottom_corners[:, 0]
                ys_bottom = bottom_corners[:, 1]
                zs_bottom = bottom_corners[:, 2]
                
                xs_top = top_corners[:, 0]
                ys_top = top_corners[:, 1]
                zs_top = top_corners[:, 2]
                
                # Draw bottom face
                ax.plot(xs_bottom, ys_bottom, zs_bottom, color='orange')
                
                # Draw top face
                ax.plot(xs_top, ys_top, zs_top, color='orange')
                
                # Draw vertical edges (connect corresponding corners)
                for i in range(len(xs_bottom)-1):  # -1 because last point closes the loop
                    ax.plot([xs_bottom[i], xs_top[i]], 
                        [ys_bottom[i], ys_top[i]], 
                        [zs_bottom[i], zs_top[i]], 
                        color='orange')
        set_axes_equal(ax)
        plt.show()

    def make_obstacles(self, gates_pos, gates_rpy, obstacles_pos, gate_size, obstacle_size):
        obstacles = [
            {'center': pos.tolist(), 'size': obstacle_size, 'rotation': [0, 0, 0]}
            for pos in obstacles_pos
        ] + [
            {'center': pos.tolist(), 'size': gate_size, 'rotation': rpy.tolist()}
            for pos, rpy in zip(gates_pos, gates_rpy)
        ]
        
        return obstacles

    def rpy_to_tangents(self, rpy_array, magnitude=1.0):
        yaw = rpy_array[:, 2] + np.pi / 2  # Adjust direction to fly into gate
        yaw[0] = yaw[0] - np.pi # first yaw entry is the drone, need to fly downwards
        dx = np.cos(yaw)
        dy = np.sin(yaw)
        dz = np.zeros_like(yaw)
        return magnitude * np.stack([dx, dy, dz], axis=1)


# Testing
# gates_pos = np.array([
#     [0.45, -0.5, 0.56],
#     [1.0, -1.05, 1.11],
#     [0.0, 1.0, 0.56],
#     [-0.5, 0.0, 1.11]
# ])

# gates_rpy = np.array([
#     [0.0, 0.0, 2.35],
#     [0.0, 0.0, -0.78],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 3.14]

# ])

# obstacles = np.array([
#     [ 0.8585742,   0.07062764,  1.430361  ],
#     [ 0.5,        -1.0,          1.4       ],
#     [ 0.,          1.5,         1.4       ],
#     [-0.5,         0.5,         1.4       ]
# ])

# pp = PathPlanner([1.0, 1.5, 0.07], gates_pos, gates_rpy, obstacles)
# # pp.update(gates_pos, gates_rpy, obstacles)
# pp.plot()
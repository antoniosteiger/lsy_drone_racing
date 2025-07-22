import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

POINTS_PER_SECTION = 100
OBSTACLE_SIZE = [0.10, 0.10, 2.0]
GATE_SIZE = [0.50, 0.03, 0.50]
CLEARANCE = 0.22
WAYPOINT_SPEED = 1.1

class PathPlanner:
    def __init__(self, pos, waypoints, tangents, obstacles_pos):
        self.initial_pos = pos

        self.waypoints = []
        self.tangents =[]
        self.obstacles = []
        self.path = []
        self.collision_indices = []
        self.detour_waypoints = []

        self.plan(pos, waypoints, tangents, obstacles_pos)

    def plan(self, pos, waypoints, tangents, obstacles_pos):
        MAX_ITERATIONS = 1
        STARTING_TANGENT = [0, 0, 0]
        END_POS = [-0.5, -0.8, 1.11]
        END_TANGENT = [0, 0, 3.14]

        self.waypoints = np.vstack([self.initial_pos, waypoints, END_POS])
        self.tangents = np.vstack([STARTING_TANGENT, tangents, END_TANGENT])
        self.obstacles = self.make_obstacles(self.waypoints, self.tangents, obstacles_pos, GATE_SIZE, OBSTACLE_SIZE)
        
        tangents = self.rpy_to_tangents(self.tangents, WAYPOINT_SPEED)
        path = self.make_path(self.waypoints, tangents)
        
        # Check for collisions and replan MAX_ITERATIONS+1 times
        for i in range(MAX_ITERATIONS):
            collision_indices, collision_obstacles = self.check_collisions(path, self.obstacles)
            if i == 0:
                self.collision_indices = collision_indices # update for plotting
                print(collision_indices)
            if len(self.collision_indices) > 0:
                self.detour_waypoints, detour_tangents = self.detour(path, self.waypoints, tangents, collision_indices, collision_obstacles)
                # print("Detour_Waypoints:\n", self.detour_waypoints)
                # print("Detour_Tangents:\n", detour_tangents)
                path = self.make_path(self.detour_waypoints, detour_tangents)
                
            if len(collision_indices) > 0:
                continue
            else:
                break

        self.path = path
        return path


    
    def make_path(self, waypoints, tangents):
        # Generate path
        
        p = np.arange(len(waypoints)) # parametrization
        if tangents is not None:
            splines = [CubicHermiteSpline(p, waypoints[:, i], tangents[:, i]) for i in range(3)]
        else:
            splines = [CubicSpline(p, waypoints[:, i]) for i in range(3)]
        p_fine = np.linspace(p[0], p[-1], (len(waypoints)-1) * POINTS_PER_SECTION)
        path = np.stack([spline(p_fine) for spline in splines], axis=1)

        self.path = path
        return path
    
    
    def check_collisions(self, path, obstacles, margin=0.08):
        all_collision_indices = []
        collision_obstacles = {}
        RESOLUTION = 30
    
        for obstacle in obstacles:
            center = np.array(obstacle['center'])
            size = np.array(obstacle['size'])
            rotation = obstacle.get('rotation', [0, 0, 0])
            
            # Add margin to obstacle size
            expanded_size = size + 2 * margin
            half_size = expanded_size / 2
            
            # Transform all path points to obstacle's local coordinate system
            relative_points = path - center
            
            # Apply inverse rotation if obstacle is rotated
            if np.any(rotation):
                rotation_matrix = R.from_euler('XYZ', rotation).as_matrix()
                local_points = relative_points @ rotation_matrix
            else:
                local_points = relative_points
            
            # Check if points are within expanded obstacle bounds (vectorized)
            within_x = np.abs(local_points[:, 0]) <= half_size[0]
            within_y = np.abs(local_points[:, 1]) <= half_size[1]
            within_z = np.abs(local_points[:, 2]) <= half_size[2]
            
            # Combine all bounds checks
            within_bounds = within_x & within_y & within_z
            
            # Get collision indices for this obstacle
            collision_indices = np.where(within_bounds)[0]
            all_collision_indices.extend(collision_indices)

            # log collision obstacles:
            for i in collision_indices:
                collision_obstacles[i] = obstacle
        
        
        # Remove duplicates and sort
        unique_collision_indices = sorted(list(set(all_collision_indices)))
        
        # Apply gate center filtering (exclude points near gate centers)
        filtered_indices = []
        for idx in unique_collision_indices:
            section_position = idx % POINTS_PER_SECTION
            # Keep points that are NOT in gate center region (between 10 and 90)
            if not (section_position >= 90 or section_position <= 10):
                filtered_indices.append(idx)
        
        # Edge detection: keep only first index from consecutive sequences
        if not filtered_indices:
            return [], []
        
        edge_indices = [filtered_indices[0]]  # Always keep the first index
        last_added = filtered_indices[0]

        for i in filtered_indices[1:]:
            if i - last_added >= RESOLUTION:
                edge_indices.append(i)
                last_added = i
        
        # update for plotting
        self.collision_indices = edge_indices
        arr1 = np.array(edge_indices, dtype=np.int64)
        print(f"Filtered Indices: {arr1}")
        return edge_indices, collision_obstacles
    
    def detour(self, path, waypoints, tangents, collision_indices, collision_obstacles):
        # Idea: Add a detour point, which is the sum of a heading vector and collision vector
        # Collision vector points from obstacle center to collision point (away from obstacle)
        # Heading vector points from obstacle center to next gate (helps decide side of obstacle to fly past)
        
        detour_waypoints = waypoints
        detour_tangents = tangents


        for i in reversed(collision_indices):
            current_section = i // POINTS_PER_SECTION
            center = collision_obstacles[i]['center']
            center[2] = path[i][2]
            
            # calculate detour point:
            heading = waypoints[current_section+1] - center
            heading[2] = 0
            heading = heading / np.linalg.norm(heading)

            collision = path[i] - center
            # collision[2] = 0
            collision = collision / np.linalg.norm(collision)

            detour_vector = (collision + heading)
            detour_xy_norm = np.linalg.norm(detour_vector[:2])
            detour_vector = detour_vector / detour_xy_norm
            detour_point = path[i] + CLEARANCE * detour_vector


            # calculate detour tangent:
            detour_tangent = waypoints[current_section+1] - detour_point
            detour_tangent = detour_tangent / np.linalg.norm(detour_tangent)

            # save:
            detour_waypoints = np.insert(detour_waypoints, current_section+1, detour_point, axis=0)
            detour_tangents = np.insert(detour_tangents, current_section+1, detour_tangent, axis=0)

            # print("Collision Index:", i)
            # print("Collision Indices:", collision_indices)
            # print("Obstacle", collision_obstacles[i])
            # print("Waypoints:", waypoints)
            # print("Current Section:", current_section)
            # print("Center:", center)
            # print("Heading:", heading)
            # print("Collision:", collision)
            # print("Detour Vector:", detour_vector)
            # print("Detour Point:", detour_point)
            # print("Detour Tangent:", detour_tangent)
            # print("Detour Waypoints:", detour_waypoints)
            # print("Detour Tangents:", detour_tangents)
            # break


        
        return detour_waypoints, detour_tangents
    

    def make_obstacles(self, waypoints, tangents, obstacles_pos, gate_size, obstacle_size):
        obstacles = [
            {'center': pos.tolist(), 'size': obstacle_size, 'rotation': [0, 0, 0]}
            for pos in obstacles_pos
        ] + [
            {'center': pos.tolist(), 'size': gate_size, 'rotation': rpy.tolist()}
            for pos, rpy in zip(waypoints, tangents) # ignore starting position
        ]
        
        return obstacles
    

    def rpy_to_tangents(self, rpy_array, magnitude=1.0):
        yaw = rpy_array[:, 2] + np.pi / 2  # Adjust direction to fly into gate
        yaw[0] = yaw[0] - np.pi # first yaw entry is the drone, need to fly downwards
        dx = np.cos(yaw)
        dy = np.sin(yaw)
        dz = np.zeros_like(yaw)
        return magnitude * np.stack([dx, dy, dz], axis=1)

        
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
        ax.view_init(elev=90, azim=-90)
        ax.plot(*self.path.T)
        
        # plot waypoints
        ax.scatter(*self.waypoints.T, color='green')
        
        # plot collision points
        if len(self.collision_indices) > 0:
            collision_points = self.path[self.collision_indices]
            # print(collision_points)
            ax.scatter(*collision_points.T, color='red')

        # plot reroute points
        if len(self.detour_waypoints) > 0:
            ax.scatter(*self.detour_waypoints.T, color='blue')
            print(self.detour_waypoints)

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


#Testing
waypoints = np.array([
    [0.45, -0.5, 0.56],
    [1.0, -1.05, 1.11],
    [0.0, 1.0, 0.56],
    [-0.5, 0.0, 1.11]
])

tangents = np.array([
    [0.0, 0.0, 2.35],
    [0.0, 0.0, -0.78],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 3.14]

])

obstacles = np.array([
    [ 0.8585742,   0.07062764,  1.430361  ],
    [ 0.5,        -1.0,          1.4       ],
    [ 0.,          1.5,         1.4       ],
    [-0.5,         0.5,         1.4       ]
])

pp = PathPlanner([1.0, 1.5, 0.07], waypoints, tangents, obstacles)
# pp.update(waypoints, tangents, obstacles)
pp.plot()
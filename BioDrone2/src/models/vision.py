import numpy as np
import torch

class FlyRetina:
    """
    Simulates a 1D compound eye with depth perception via optic flow/intensity.
    Transforms 3D geometry into a visual signal (brightness/depth array).
    """
    def __init__(self, num_photoreceptors=32, fov_deg=120):
        self.res = num_photoreceptors
        self.fov = np.deg2rad(fov_deg)
        # Ray angles relative to heading (0 is straight ahead)
        self.angles = np.linspace(-self.fov/2, self.fov/2, self.res)
        self.max_depth = 10.0 # Max vision range (meters)

    def render(self, drone_pos, drone_yaw, tunnel_width=6.0, obstacles=[]):
        """
        Raycasts against walls and obstacles.
        
        Args:
            drone_pos: [x, y, z]
            drone_yaw: float (radians) relative to world X-axis
            tunnel_width: Width of the corridor (walls at +/- width/2)
            obstacles: List of boxes [x_min, x_max, y_min, y_max, z_min, z_max]
        
        Returns:
            intensity: (num_photoreceptors,) array. 1.0=Close, 0.0=Far.
        """
        
        # 1. Compute Ray Directions in World Space
        ray_angles_world = drone_yaw + self.angles
        ray_dx = np.cos(ray_angles_world)
        ray_dy = np.sin(ray_angles_world)
        
        drone_x, drone_y, drone_z = drone_pos
        
        # 2. Raycast against Walls (Infinite planes at Y = +/- width/2)
        y_min, y_max = -tunnel_width / 2.0, tunnel_width / 2.0
        
        # Initialize with max depth
        t_final = np.full(self.res, self.max_depth)
        
        # Wall Intersection (Slab method simplifed for infinite planes)
        inv_dy = 1.0 / (ray_dy + 1e-6) 
        t_wall_1 = (y_min - drone_y) * inv_dy
        t_wall_2 = (y_max - drone_y) * inv_dy
        
        # Select the Positive t (forward intersection)
        t_walls = np.full(self.res, self.max_depth)
        t_walls[ray_dy > 0] = t_wall_2[ray_dy > 0] # Looking Left/+Y hits Max Wall
        t_walls[ray_dy < 0] = t_wall_1[ray_dy < 0] # Looking Right/-Y hits Min Wall
        
        t_walls[t_walls < 0] = self.max_depth # Clip backward hits
        t_final = np.minimum(t_final, t_walls)

        # 3. Raycast against Obstacles (Box Intersection)
        for box in obstacles:
            bx_min, bx_max, by_min, by_max, bz_min, bz_max = box
            
            # Simple Z height check (2.5D vision optimization)
            if not (bz_min <= drone_z <= bz_max):
                continue
                
            # X-Slab
            inv_dx = 1.0 / (ray_dx + 1e-6)
            t1_x = (bx_min - drone_x) * inv_dx
            t2_x = (bx_max - drone_x) * inv_dx
            t_enter_x = np.minimum(t1_x, t2_x)
            t_exit_x = np.maximum(t1_x, t2_x)
            
            # Y-Slab
            t1_y = (by_min - drone_y) * inv_dy
            t2_y = (by_max - drone_y) * inv_dy
            t_enter_y = np.minimum(t1_y, t2_y)
            t_exit_y = np.maximum(t1_y, t2_y)
            
            # Intersection
            t_enter = np.maximum(t_enter_x, t_enter_y)
            t_exit = np.minimum(t_exit_x, t_exit_y)
            
            # Hit? (Enter < Exit) and (Exit > 0)
            hit_mask = (t_enter < t_exit) & (t_exit > 0)
            
            # Update minimums
            dist = np.maximum(0, t_enter)
            t_final = np.where(hit_mask, np.minimum(t_final, dist), t_final)

        # 4. Invert to Intensity (1.0 = Touching/Collision)
        intensity = 1.0 - (np.clip(t_final, 0, self.max_depth) / self.max_depth)
        
        return intensity.astype(np.float32)
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Ensure import of Vision module
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.vision import FlyRetina

class Drone3DEnv(gym.Env):
    """
    Phase F (Hard Mode): 3D Drone Navigation with inescapable obstacles.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = 1000 
        
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.left_eye = FlyRetina(num_photoreceptors=32, fov_deg=100)
        self.right_eye = FlyRetina(num_photoreceptors=32, fov_deg=100)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32
        )
        
        self.dt = 0.05
        self.obstacles = []

    def _generate_obstacle(self, x_pos):
        """Helper to create tricky obstacles"""
        # Type 1: The Pillar (covers a vertical slice)
        # Type 2: The Bar (covers a horizontal slice)
        # Type 3: The Wall (covers almost everything, requires precise hole)
        
        obs_type = self.np_random.choice(['pillar', 'bar', 'wall'], p=[0.6, 0.3, 0.1])
        
        if obs_type == 'pillar':
            # Spawn anywhere across full width (-3 to 3)
            # Make sure it's not ALWAYS center
            y = self.np_random.uniform(-2.5, 2.5) 
            w = self.np_random.uniform(0.8, 1.5) # Fatter pillars
            h = self.np_random.uniform(1.0, 3.0)
            z_base = 0.0
            if self.np_random.random() > 0.5: # Floating pillar
                z_base = self.np_random.uniform(0.5, 2.0)
            return [x_pos, x_pos + w, y - w/2, y + w/2, z_base, z_base + h]

        elif obs_type == 'bar':
             # Horizontal bar blocking the path
             y_center = 0.0
             w_y = self.np_random.uniform(4.0, 6.0) # Blocks almost full width (6m)
             depth = 0.5
             h = 0.5 # Thin bar
             z = self.np_random.uniform(0.5, 2.5) # Random height
             return [x_pos, x_pos + depth, -w_y/2, w_y/2, z, z + h]
             
        elif obs_type == 'wall':
            # A wall with a gap? Or just a huge block on one side?
            # Let's do a "Half-Wall" forcing strict side selection
            side = 1 if self.np_random.random() > 0.5 else -1
            # Blocks from -3 to 0.5 OR -0.5 to 3
            y_min = -3.0 if side == -1 else -0.5
            y_max = 0.5 if side == -1 else 3.0
            return [x_pos, x_pos + 1.0, y_min, y_max, 0.0, 3.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(9, dtype=np.float32)
        self.state[2] = 1.0 
        self.state[0:3] += self.np_random.uniform(-0.1, 0.1, 3)
        self.step_count = 0
        
        self.obstacles = []
        # Initial easy zone
        for i in range(20): 
            x = 10.0 + (i * 6.0) # slightly larger spacing to allow maneuvering
            self.obstacles.append(self._generate_obstacle(x))
            
        return self._get_obs(), {}

    def step(self, action):
        pos = self.state[0:3]
        vel = self.state[3:6]
        
        roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd = action
        
        accel = np.array([
            pitch_cmd * 5.0, 
            -roll_cmd * 5.0,
            (thrust_cmd * 15.0) - 2.0
        ])
        
        vel += (accel - 0.1 * vel) * self.dt 
        pos += vel * self.dt
        
        self.state[0:3] = pos
        self.state[3:6] = vel
        self.state[6:9] = [roll_cmd, pitch_cmd, yaw_cmd]
        
        self.step_count += 1
        
        # --- Procedural Generation (Infinite) ---
        self.obstacles = [o for o in self.obstacles if o[1] > pos[0] - 5.0]
        while len(self.obstacles) < 20:
            last_x = self.obstacles[-1][1] if self.obstacles else pos[0]
            # Randomize gap slightly (4m to 7m)
            dist = self.np_random.uniform(4.0, 7.0)
            self.obstacles.append(self._generate_obstacle(last_x + dist))

        # --- Collision ---
        crashed = False
        if pos[1] < -3.0 or pos[1] > 3.0 or pos[2] < 0.2 or pos[2] > 3.0: 
            crashed = True
            
        drone_r = 0.2
        for box in self.obstacles:
             if (pos[0] + drone_r > box[0] and pos[0] - drone_r < box[1] and
                 pos[1] + drone_r > box[2] and pos[1] - drone_r < box[3] and
                 pos[2] + drone_r > box[4] and pos[2] - drone_r < box[5]):
                 crashed = True
                 break

        terminated = crashed
        truncated = self.step_count >= self.max_steps
        
        reward = 1.0 + (vel[0] * 0.1) 
        if crashed: reward = -20.0 # Increased penalty to discourage "scraping"
        
        return self._get_obs(), reward, terminated, truncated, {}
        
    def _get_obs(self):
        vis_L = self.left_eye.render(self.state[0:3], np.deg2rad(45), obstacles=self.obstacles)
        vis_R = self.right_eye.render(self.state[0:3], np.deg2rad(-45), obstacles=self.obstacles)
        return np.concatenate([vis_L, vis_R, self.state[3:9]]).astype(np.float32)
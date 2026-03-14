"""
env_simple.py — Simple 2.5D tunnel navigation (no gates yet)
Just learn to fly straight and avoid walls first.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleTunnelEnv(gym.Env):
    metadata = {'render_modes': []}
    
    SENSOR_ANGLES_H = [-60, -30, 0, 30, 60]  # 5 horizontal sensors
    
    def __init__(self, max_steps=1000, tunnel_hw=70.0, tunnel_hh=80.0):
        super().__init__()
        self.max_steps = max_steps
        self.base_hw = tunnel_hw
        self.base_hh = tunnel_hh
        
        # 7 sensors: 5 horizontal + up + down
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        
        # 5 actions: LEFT, FWD, RIGHT, UP, DOWN
        self.action_space = spaces.Discrete(5)
        
        self.sensor_max_range = 150.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_vx = 0.0
        self.drone_vy = 0.0
        
        self.tunnel_cx = 0.0
        self.tunnel_cy = 0.0
        self.hw = float(self.base_hw)
        self.hh = float(self.base_hh)
        
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        
        # Apply action (speed = 9.0)
        speed = 9.0
        if action == 0:    # LEFT
            self.drone_vx = -speed
            self.drone_vy = 0.0
        elif action == 1:  # FWD (stay centered)
            self.drone_vx = 0.0
            self.drone_vy = 0.0
        elif action == 2:  # RIGHT
            self.drone_vx = speed
            self.drone_vy = 0.0
        elif action == 3:  # UP
            self.drone_vx = 0.0
            self.drone_vy = -speed
        elif action == 4:  # DOWN
            self.drone_vx = 0.0
            self.drone_vy = speed
        
        # Apply friction
        self.drone_vx *= 0.85
        self.drone_vy *= 0.85
        
        # Clamp velocity
        max_v = 12.0
        self.drone_vx = np.clip(self.drone_vx, -max_v, max_v)
        self.drone_vy = np.clip(self.drone_vy, -max_v, max_v)
        
        # Update position
        self.drone_x += self.drone_vx
        self.drone_y += self.drone_vy
        
        # Get observation
        obs = self._get_obs()
        
        # Check wall collision
        left = self.tunnel_cx - self.hw
        right = self.tunnel_cx + self.hw
        top = self.tunnel_cy - self.hh
        bot = self.tunnel_cy + self.hh
        
        hit_wall = (self.drone_x < left or self.drone_x > right or
                    self.drone_y < top or self.drone_y > bot)
        
        # Reward shaping
        if hit_wall:
            reward = -100.0
            terminated = True
        else:
            # Reward for staying centered and moving forward
            dx = abs(self.drone_x - self.tunnel_cx)
            dy = abs(self.drone_y - self.tunnel_cy)
            center_reward = 0.5 * (1.0 - dx / self.hw) + 0.3 * (1.0 - dy / self.hh)
            reward = 1.0 + center_reward
            terminated = False
        
        truncated = self.step_count >= self.max_steps
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        obs = []
        
        # 5 horizontal sensors
        left = self.tunnel_cx - self.hw
        right = self.tunnel_cx + self.hw
        top = self.tunnel_cy - self.hh
        bot = self.tunnel_cy + self.hh
        
        for angle_deg in self.SENSOR_ANGLES_H:
            angle_rad = np.radians(angle_deg)
            dx = np.sin(angle_rad)
            dy = -np.cos(angle_rad)
            
            # Ray cast to walls
            if dx < -1e-6:
                d_left = (self.drone_x - left) / (-dx)
            elif dx > 1e-6:
                d_left = (right - self.drone_x) / dx
            else:
                d_left = 999.0
            
            if dy < -1e-6:
                d_top = (self.drone_y - top) / (-dy)
            elif dy > 1e-6:
                d_top = (bot - self.drone_y) / dy
            else:
                d_top = 999.0
            
            d = min(d_left, d_top, self.sensor_max_range)
            obs.append(d / self.sensor_max_range)
        
        # UP sensor
        d_up = self.drone_y - top
        obs.append(min(d_up, self.sensor_max_range) / self.sensor_max_range)
        
        # DOWN sensor
        d_down = bot - self.drone_y
        obs.append(min(d_down, self.sensor_max_range) / self.sensor_max_range)
        
        return np.array(obs, dtype=np.float32)
    
    def close(self):
        pass
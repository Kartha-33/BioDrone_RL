import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Ensure we can import from sibling directories
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from models.vision import FlyRetina

class TunnelEnv(gym.Env):
    """
    Phase B: Vision-Based Tunnel Environment
    Goal: Navigate using only 1D visual input (Retina).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.width = 2.0
        self.max_steps = 200
        
        # Action: 0=No-Op, 1=Left force, 2=Right force
        self.action_space = spaces.Discrete(3)
        
        # Vision Setup
        self.retina = FlyRetina(num_photoreceptors=32, fov_deg=120)
        
        # Observation: 32-pixel intensity vector [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(32,), 
            dtype=np.float32
        )
        
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start near center with small random velocity
        self.pos = self.np_random.uniform(-0.2, 0.2)
        self.vel = self.np_random.uniform(-0.1, 0.1)
        
        self.step_count = 0
        
        # Generate initial visual observation
        obs = self.retina.render(self.pos, self.width)
        return obs, {}

    def step(self, action):
        force = 0.0
        if action == 1: force = -0.1
        elif action == 2: force = 0.1
        
        # Dynamics
        self.vel += force
        self.vel *= 0.95  # Damping simulation (air resistance)
        self.pos += self.vel
        
        self.step_count += 1
        
        # Check termination
        # Wall is at -1.0 and 1.0. We crash if we touch them.
        crashed = bool(self.pos < -1.0 or self.pos > 1.0)
        terminated = crashed
        truncated = self.step_count >= self.max_steps
        
        # Improved Reward Function for Vision
        # 1. Survival Reward: +1.0
        # 2. Centering Reward: Higher if closer to 0.0 (using Gaussian)
        centering_bonus = np.exp(-5.0 * (self.pos**2))  # Max 1.0 at center
        
        reward = 1.0 + centering_bonus
        if crashed:
            reward = -10.0
        
        # Generate new observation
        obs = self.retina.render(self.pos, self.width)
        
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass # Still a placeholder until Phase B part 2

    def close(self):
        pass
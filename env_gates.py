"""
env_gates.py — Tunnel navigation WITH gates (clean version)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Gate:
    def __init__(self, dist, cx, cy, gap_w, gap_h):
        self.dist_ahead = dist
        self.center_x = cx
        self.center_y = cy
        self.gap_width = gap_w
        self.gap_height = gap_h
        self.passed = False
    
    def is_drone_through(self, drone_x, drone_y):
        """Check if drone passed through gate opening cleanly"""
        if abs(self.dist_ahead) > 2:
            return False
        in_x = abs(drone_x - self.center_x) < self.gap_width / 2
        in_y = abs(drone_y - self.center_y) < self.gap_height / 2
        return in_x and in_y
    
    def is_drone_hit_frame(self, drone_x, drone_y):
        """Check if drone hit the gate frame (not the opening)"""
        if abs(self.dist_ahead) > 3:
            return False
        
        # Check if safely inside the gap
        in_x = abs(drone_x - self.center_x) < self.gap_width / 2
        in_y = abs(drone_y - self.center_y) < self.gap_height / 2
        
        if in_x and in_y:
            return False  # Inside gap = safe
        
        # Check if near gate at all (within frame bounds)
        frame_w = self.gap_width + 15.0
        frame_h = self.gap_height + 15.0
        near_x = abs(drone_x - self.center_x) < frame_w / 2
        near_y = abs(drone_y - self.center_y) < frame_h / 2
        
        # Hit frame only if near gate but not through opening
        return near_x and near_y

class GateTunnelEnv(gym.Env):
    metadata = {'render_modes': []}
    
    SENSOR_ANGLES_H = [-60, -30, 0, 30, 60]
    
    def __init__(self, max_steps=1000, tunnel_hw=70.0, tunnel_hh=80.0, gate_every=20):
        super().__init__()
        self.max_steps = max_steps
        self.base_hw = tunnel_hw
        self.base_hh = tunnel_hh
        self.gate_spacing = gate_every
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
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
        
        self.gates = []
        self._spawn_gates()
        self.gates_passed = 0
        
        obs = self._get_obs()
        return obs, {}
    
    def _spawn_gates(self):
        """Spawn 10 gates ahead"""
        for i in range(10):
            dist = (i + 1) * self.gate_spacing
            
            # Gate centered in tunnel with some random offset
            cx = self.tunnel_cx + np.random.uniform(-self.hw * 0.3, self.hw * 0.3)
            cy = self.tunnel_cy + np.random.uniform(-self.hh * 0.3, self.hh * 0.3)
            
            # Gap size: 50-70% of tunnel size
            gap_w = np.random.uniform(self.hw * 0.5, self.hw * 0.7)
            gap_h = np.random.uniform(self.hh * 0.5, self.hh * 0.7)
            
            self.gates.append(Gate(dist, cx, cy, gap_w, gap_h))
    
    def step(self, action):
        self.step_count += 1
        
        # Apply action
        speed = 9.0
        if action == 0:    # LEFT
            self.drone_vx = -speed
            self.drone_vy = 0.0
        elif action == 1:  # FWD
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
        
        # Physics
        self.drone_vx *= 0.85
        self.drone_vy *= 0.85
        self.drone_vx = np.clip(self.drone_vx, -12.0, 12.0)
        self.drone_vy = np.clip(self.drone_vy, -12.0, 12.0)
        
        self.drone_x += self.drone_vx
        self.drone_y += self.drone_vy
        
        # Advance gates (tunnel scrolls forward)
        for gate in self.gates:
            gate.dist_ahead -= 1
        
        # Check wall collision
        left = self.tunnel_cx - self.hw
        right = self.tunnel_cx + self.hw
        top = self.tunnel_cy - self.hh
        bot = self.tunnel_cy + self.hh
        
        hit_wall = (self.drone_x < left or self.drone_x > right or
                    self.drone_y < top or self.drone_y > bot)
        
        # Check gates
        gate_collision = False
        gate_reward = 0.0
        
        for gate in self.gates:
            if gate.passed:
                continue
            
            if gate.is_drone_hit_frame(self.drone_x, self.drone_y):
                gate_collision = True
                break
            
            if gate.is_drone_through(self.drone_x, self.drone_y):
                if not gate.passed:
                    gate.passed = True
                    self.gates_passed += 1
                    # Big bonus for passing through center
                    dx = abs(self.drone_x - gate.center_x) / gate.gap_width
                    dy = abs(self.drone_y - gate.center_y) / gate.gap_height
                    center_bonus = 1.0 - (dx + dy) / 2.0
                    gate_reward = 100.0 + 50.0 * center_bonus
        
        # Spawn new gate when furthest one gets close
        if self.gates and self.gates[-1].dist_ahead < 100:
            new_dist = self.gates[-1].dist_ahead + self.gate_spacing
            cx = self.tunnel_cx + np.random.uniform(-self.hw * 0.3, self.hw * 0.3)
            cy = self.tunnel_cy + np.random.uniform(-self.hh * 0.3, self.hh * 0.3)
            gap_w = np.random.uniform(self.hw * 0.5, self.hw * 0.7)
            gap_h = np.random.uniform(self.hh * 0.5, self.hh * 0.7)
            self.gates.append(Gate(new_dist, cx, cy, gap_w, gap_h))
        
        # Remove passed gates
        self.gates = [g for g in self.gates if g.dist_ahead > -10]
        
        # Compute reward
        if hit_wall or gate_collision:
            reward = -100.0
            terminated = True
        else:
            dx = abs(self.drone_x - self.tunnel_cx)
            dy = abs(self.drone_y - self.tunnel_cy)
            center_reward = 0.5 * (1.0 - dx / self.hw) + 0.3 * (1.0 - dy / self.hh)
            reward = 1.0 + center_reward + gate_reward
            terminated = False
        
        truncated = self.step_count >= self.max_steps
        info = {'gates_passed': self.gates_passed}
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        obs = []
        
        left = self.tunnel_cx - self.hw
        right = self.tunnel_cx + self.hw
        top = self.tunnel_cy - self.hh
        bot = self.tunnel_cy + self.hh
        
        for angle_deg in self.SENSOR_ANGLES_H:
            angle_rad = np.radians(angle_deg)
            dx = np.sin(angle_rad)
            dy = -np.cos(angle_rad)
            
            # Ray to walls
            if dx < -1e-6:
                d_wall = (self.drone_x - left) / (-dx)
            elif dx > 1e-6:
                d_wall = (right - self.drone_x) / dx
            else:
                d_wall = 999.0
            
            if dy < -1e-6:
                d_wall = min(d_wall, (self.drone_y - top) / (-dy))
            elif dy > 1e-6:
                d_wall = min(d_wall, (bot - self.drone_y) / dy)
            
            # Ray to gates
            for gate in self.gates:
                if 0 < gate.dist_ahead < 50:
                    # Simple gate detection
                    gate_dist = gate.dist_ahead / np.cos(angle_rad) if abs(np.cos(angle_rad)) > 0.1 else 999
                    d_wall = min(d_wall, gate_dist)
            
            d = min(d_wall, self.sensor_max_range)
            obs.append(d / self.sensor_max_range)
        
        # UP/DOWN sensors
        d_up = self.drone_y - top
        d_down = bot - self.drone_y
        obs.append(min(d_up, self.sensor_max_range) / self.sensor_max_range)
        obs.append(min(d_down, self.sensor_max_range) / self.sensor_max_range)
        
        return np.array(obs, dtype=np.float32)
    
    def close(self):
        pass
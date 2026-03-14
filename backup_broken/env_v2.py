"""
env_v2.py — BioDrone 2.5D Environment with Gates and Obstacles

Upgrades from env.py:
  - Adds vertical dimension (drone can move up/down)
  - Gate obstacles at varying heights and widths
  - Drone must fly THROUGH gates (not just avoid side walls)
  - 7 sensors: 5 horizontal + 2 vertical (up/down)
  - Action space: 5 actions (Left, Right, Up, Down, Straight)
  - Reward shaping: bonus for flying through gate centres
  - Compatible with existing MaskedLinear model (just bigger input/output)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Gate:
    """A rectangular gate obstacle the drone must fly through."""

    def __init__(self, dist_ahead, center_x, center_y,
                 gap_width, gap_height, tunnel_half_w):
        """
        dist_ahead    : how far ahead of drone this gate is (env steps)
        center_x      : horizontal centre of gate opening
        center_y      : vertical centre of gate opening  
        gap_width     : width of the gate opening
        gap_height    : height of the gate opening
        tunnel_half_w : half-width of enclosing tunnel (for bounds)
        """
        self.dist_ahead    = dist_ahead
        self.center_x      = center_x
        self.center_y      = center_y
        self.gap_width     = gap_width
        self.gap_height    = gap_height
        self.tunnel_half_w = tunnel_half_w
        self.passed        = False

    def drone_collides(self, drone_x, drone_y):
        """Check if drone hits the GATE FRAME (not the gap)"""
        if abs(self.dist_ahead) > 2:
            return False
        
        # Check if inside the gap (safe zone)
        in_x = abs(drone_x - self.center_x) < self.gap_width  / 2
        in_y = abs(drone_y - self.center_y) < self.gap_height / 2
        
        # If inside gap → no collision
        if in_x and in_y:
            return False
        
        # If outside tunnel bounds at gate position → hit frame
        # (only check collision if drone is AT the gate, not before/after)
        return True  # ← BUG: always returns True if not in gap!

    def drone_passed_through(self, drone_x, drone_y):
        """Returns True if drone cleanly passed through gate opening."""
        in_x = abs(drone_x - self.center_x) < self.gap_width  / 2
        in_y = abs(drone_y - self.center_y) < self.gap_height / 2
        return in_x and in_y


class BioDroneEnv2(gym.Env):
    """
    2.5D drone environment with:
      - Curving tunnel (horizontal)
      - Vertical movement
      - Gate obstacles at varying heights
      - 7-sensor observation space
      - 5 discrete actions
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Sensor angles for 5 horizontal rays (degrees, 0=forward)
    SENSOR_ANGLES_H = [-60, -25, 0, 25, 60]
    # 2 vertical sensors
    SENSOR_ANGLES_V = [-90, 90]   # up, down

    def __init__(self, max_steps=1000, tunnel_half_w=70,
                 tunnel_half_h=80, gate_every=15, difficulty=1.0):
        super().__init__()

        self.max_steps      = max_steps
        self.base_half_w    = tunnel_half_w
        self.base_half_h    = tunnel_half_h
        self.gate_every     = gate_every    # steps between gates
        self.difficulty     = difficulty    # 1.0 = normal, 2.0 = hard

        # 7 sensor readings: 5 horizontal + 2 vertical
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # 5 actions: Left, Right, Up, Down, Straight
        self.action_space = spaces.Discrete(5)

        # Sensor properties
        self.sensor_angles_deg  = self.SENSOR_ANGLES_H
        self.sensor_max_range   = 150.0

        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random if seed is None else np.random.default_rng(seed)

        self.step_count      = 0
        self.drone_x         = 0.0       # lateral position
        self.drone_y         = 0.0       # vertical position
        self.drone_vx        = 0.0       # lateral velocity
        self.drone_vy        = 0.0       # vertical velocity

        # Tunnel geometry
        self.tunnel_center_x = 0.0
        self.tunnel_center_y = 0.0
        self.half_width      = float(self.base_half_w)
        self.half_height     = float(self.base_half_h)

        # Tunnel drift (random walk)
        self.drift_x         = 0.0
        self.drift_y         = 0.0

        # Gates queue: list of Gate objects
        self.gates           = []
        self._spawn_initial_gates()

        obs = self._get_obs()
        return obs, {}

    # ── Gates ─────────────────────────────────────────────────────────────

    def _spawn_gate(self, dist_ahead):
        """Spawn a gate at dist_ahead steps ahead of drone."""
        rng = self.np_random

        # Gate centre: somewhere in the tunnel, at a random height
        cx = self.tunnel_center_x + rng.uniform(
            -self.half_width  * 0.4,
             self.half_width  * 0.4
        )
        cy = self.tunnel_center_y + rng.uniform(
            -self.half_height * 0.5,
             self.half_height * 0.5
        )

        # Gate opening size: harder = smaller opening
        gap_w = rng.uniform(
            self.half_width  * (0.3 / (self.difficulty + 0.5)),
            self.half_width  * (0.7 / (self.difficulty + 0.2))
        )
        gap_h = rng.uniform(
            self.half_height * (0.4 / self.difficulty),
            self.half_height * (0.9 / self.difficulty)
        )
        gap_w = max(25.0, min(gap_w, self.half_width  * 1.6))
        gap_h = max(20.0, min(gap_h, self.half_height * 1.4))

        return Gate(
            dist_ahead    = dist_ahead,
            center_x      = cx,
            center_y      = cy,
            gap_width     = gap_w,
            gap_height    = gap_h,
            tunnel_half_w = self.half_width
        )

    def _spawn_initial_gates(self):
        """Spawn first few gates ahead of start."""
        for i in range(1, 6):
            self.gates.append(self._spawn_gate(i * self.gate_every))

    # ── Step ──────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Actions:
          0 = Steer Left
          1 = Steer Right
          2 = Move Up
          3 = Move Down
          4 = Straight (no lateral/vertical change)
        """
        self.step_count += 1

        # ── Apply action ──────────────────────────────────────────────
        if action == 0:        # LEFT  → move left (X decreases)
            self.drone_vx = -9.0
            self.drone_vy =  0.0
        elif action == 1:      # FWD   → stay centred, tunnel scrolls
            self.drone_vx =  0.0
            self.drone_vy =  0.0
        elif action == 2:      # RIGHT → move right (X increases)
            self.drone_vx =  9.0
            self.drone_vy =  0.0
        elif action == 3:      # UP    → move up (Y decreases)
            self.drone_vx =  0.0
            self.drone_vy = -9.0
        elif action == 4:      # DOWN  → move down (Y increases)
            self.drone_vx =  0.0
            self.drone_vy =  9.0

        # Apply friction
        friction = 0.85
        self.drone_vx *= friction
        self.drone_vy *= friction

        # Clamp velocity
        max_v = 12.0
        self.drone_vx = np.clip(self.drone_vx, -max_v, max_v)
        self.drone_vy = np.clip(self.drone_vy, -max_v, max_v)

        # Update position
        self.drone_x += self.drone_vx
        self.drone_y += self.drone_vy

        # ── Update tunnel ─────────────────────────────────────────────
        self._update_tunnel()

        # ── Advance gates ─────────────────────────────────────────────
        for gate in self.gates:
            gate.dist_ahead -= 1

        # Remove gates that have passed
        self.gates = [g for g in self.gates if g.dist_ahead > -3]

        # Spawn new gates to keep 5 ahead
        gates_ahead = [g for g in self.gates if g.dist_ahead > 0]
        while len(gates_ahead) < 5:
            furthest = max((g.dist_ahead for g in self.gates), default=0)
            self.gates.append(
                self._spawn_gate(furthest + self.gate_every)
            )
            gates_ahead = [g for g in self.gates if g.dist_ahead > 0]

        # ── Check collisions ──────────────────────────────────────────
        terminated = False
        reward     = 1.0    # survival reward

        # Wall collision (horizontal)
        left_wall  = self.tunnel_center_x - self.half_width
        right_wall = self.tunnel_center_x + self.half_width
        top_wall   = self.tunnel_center_y - self.half_height
        bot_wall   = self.tunnel_center_y + self.half_height

        if (self.drone_x < left_wall  or self.drone_x > right_wall or
                self.drone_y < top_wall   or self.drone_y > bot_wall):
            terminated = True
            reward     = -100.0
            return self._get_obs(), reward, terminated, False, \
                   {"crash": "wall"}

        # Gate collision / passage check
        for gate in self.gates:
            if abs(gate.dist_ahead) <= 2:
                if gate.drone_collides(self.drone_x, self.drone_y):
                    terminated = True
                    reward     = -100.0
                    return self._get_obs(), reward, terminated, False, \
                           {"crash": "gate"}
                elif (not gate.passed and
                      gate.drone_passed_through(self.drone_x, self.drone_y)):
                    gate.passed = True
                    # Bonus: how centered was the pass?
                    dx      = abs(self.drone_x - gate.center_x)
                    dy      = abs(self.drone_y - gate.center_y)
                    center  = 1.0 - (dx / gate.gap_width +
                                     dy / gate.gap_height) / 2.0
                    reward += 20.0 + 30.0 * center   # up to +50 per gate!

        # Centering bonus
        dx_centre = abs(self.drone_x - self.tunnel_center_x)
        dy_centre = abs(self.drone_y - self.tunnel_center_y)
        centre_bonus = max(0.0,
            0.5 * (1.0 - dx_centre / self.half_width) +
            0.3 * (1.0 - dy_centre / self.half_height)
        )
        reward += centre_bonus

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, \
               {"gates_passed": sum(g.passed for g in self.gates)}

    # ── Tunnel Update ─────────────────────────────────────────────────────

    def _update_tunnel(self):
        """Random-walk tunnel drift in X and Y."""
        rng = self.np_random

        # Horizontal drift
        self.drift_x += rng.uniform(-0.4, 0.4)
        self.drift_x  = np.clip(self.drift_x, -2.5, 2.5)
        self.tunnel_center_x += self.drift_x
        self.tunnel_center_x  = np.clip(
            self.tunnel_center_x, -80.0, 80.0
        )

        # Vertical drift (slower)
        self.drift_y += rng.uniform(-0.2, 0.2)
        self.drift_y  = np.clip(self.drift_y, -1.5, 1.5)
        self.tunnel_center_y += self.drift_y
        self.tunnel_center_y  = np.clip(
            self.tunnel_center_y, -40.0, 40.0
        )

    # ── Observation ───────────────────────────────────────────────────────

    def _get_obs(self):
        """
        7 ray-cast sensors:
          [0-4] : 5 horizontal rays at angles [-60,-25,0,25,60]
          [5]   : 1 upward ray
          [6]   : 1 downward ray

        Raw distance normalised to [0, 1] where:
          1.0 = max range (no obstacle)
          0.0 = touching wall/gate
        """
        obs = np.ones(7, dtype=np.float32)

        left_wall  = self.tunnel_center_x - self.half_width
        right_wall = self.tunnel_center_x + self.half_width
        top_wall   = self.tunnel_center_y - self.half_height
        bot_wall   = self.tunnel_center_y + self.half_height

        # ── Horizontal sensors ────────────────────────────────────────
        for i, angle in enumerate(self.SENSOR_ANGLES_H):
            a_rad  = np.radians(angle)
            dx     = np.sin(a_rad)
            # Cast ray, find distance to walls
            if dx < -1e-6:
                d = (self.drone_x - left_wall)  / (-dx)
            elif dx > 1e-6:
                d = (right_wall - self.drone_x) / dx
            else:
                d = self.sensor_max_range

            # Check if any gate intersects this ray
            for gate in self.gates:
                if 0 < gate.dist_ahead < self.sensor_max_range:
                    # Simplified: treat gate frame as a wall segment
                    gate_d = gate.dist_ahead * 2.0
                    if gate_d < d:
                        # Only blocks if ray points at gate frame
                        ray_x = self.drone_x + dx * gate_d
                        in_gap_x = abs(ray_x - gate.center_x) < \
                                   gate.gap_width / 2
                        if not in_gap_x:
                            d = gate_d

            obs[i] = float(np.clip(d / self.sensor_max_range, 0.0, 1.0))

        # ── Vertical sensors ──────────────────────────────────────────
        # Up
        d_up           = self.drone_y - top_wall
        obs[5]         = float(np.clip(
            d_up / self.sensor_max_range, 0.0, 1.0
        ))
        # Down
        d_down         = bot_wall - self.drone_y
        obs[6]         = float(np.clip(
            d_down / self.sensor_max_range, 0.0, 1.0
        ))

        return obs

    def close(self):
        pass
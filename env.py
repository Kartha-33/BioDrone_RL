import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BioDroneEnv(gym.Env):
    """
    A 2D tunnel navigation environment for the BioDrone-RL project.

    The drone moves forward at a constant speed through a tunnel that
    curves randomly. It uses 5 distance sensors (ray-casts) to perceive
    the walls around it.

    Observation: 1D numpy array of size 5 (normalized sensor distances)
        - Sensor 0: Far Left
        - Sensor 1: Near Left
        - Sensor 2: Forward
        - Sensor 3: Near Right
        - Sensor 4: Far Right

    Actions: Discrete(3)
        - 0: Steer Left
        - 1: Go Straight
        - 2: Steer Right

    Reward:
        - +1 for every step survived
        - -100 for crashing into a wall
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # --- Environment Parameters ---
        self.tunnel_width = 100.0       # Full width of the tunnel
        self.half_width = self.tunnel_width / 2.0
        self.drone_speed = 5.0          # Forward speed per step
        self.steer_speed = 4.0          # Lateral movement per step
        self.max_sensor_dist = 80.0     # Max distance a sensor can detect
        self.max_steps = 1000           # Max steps before episode truncates

        # Sensor angles relative to drone heading (in degrees)
        # [Far Left, Near Left, Forward, Near Right, Far Right]
        self.sensor_angles_deg = [-60.0, -30.0, 0.0, 30.0, 60.0]

        # --- Tunnel Curve Parameters ---
        self.curve_intensity = 0.3      # How sharply the tunnel center drifts
        self.curve_smoothness = 0.05    # Controls how gradually the curve changes

        # --- Render ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # --- State Variables (initialized in reset) ---
        self.drone_x = None             # Drone's lateral position
        self.tunnel_center_x = None     # Current tunnel center X
        self.tunnel_drift = None        # Current drift velocity of tunnel center
        self.step_count = None

    # ------------------------------------------------------------------
    # Core Gymnasium Methods
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Drone starts at the center of the tunnel
        self.drone_x = 0.0
        self.tunnel_center_x = 0.0
        self.tunnel_drift = 0.0
        self.step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # --- Apply Action: Steer the drone laterally ---
        if action == 0:   # Steer Left
            self.drone_x -= self.steer_speed
        elif action == 2: # Steer Right
            self.drone_x += self.steer_speed
        # action == 1: Go Straight — no lateral movement

        # --- Update Tunnel Curve ---
        # The tunnel center drifts smoothly using a random-walk on the drift
        noise = self.np_random.uniform(-self.curve_intensity, self.curve_intensity)
        self.tunnel_drift = (
            self.tunnel_drift * (1.0 - self.curve_smoothness)
            + noise * self.curve_smoothness
        )
        # Clamp drift so the tunnel doesn't fly off entirely
        self.tunnel_drift = np.clip(self.tunnel_drift, -3.0, 3.0)
        self.tunnel_center_x += self.tunnel_drift

        # Keep tunnel center from drifting too far from origin over time
        self.tunnel_center_x = np.clip(self.tunnel_center_x, -60.0, 60.0)

        self.step_count += 1

        # --- Check Collision ---
        left_wall = self.tunnel_center_x - self.half_width
        right_wall = self.tunnel_center_x + self.half_width

        crashed = (self.drone_x <= left_wall) or (self.drone_x >= right_wall)

        # --- Reward ---
        if crashed:
            reward = -100.0
            terminated = True
        else:
            reward = 1.0
            terminated = False

        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "drone_x": self.drone_x,
            "tunnel_center_x": self.tunnel_center_x,
            "left_wall": left_wall,
            "right_wall": right_wall,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # ------------------------------------------------------------------
    # Observation Helper
    # ------------------------------------------------------------------

    def _get_obs(self):
        """
        Cast 5 rays from the drone's position at different angles.
        Each sensor returns a normalized distance [0, 1] to the nearest wall.
        A value near 0 means the wall is very close; 1 means it's far away.
        """
        left_wall = self.tunnel_center_x - self.half_width
        right_wall = self.tunnel_center_x + self.half_width

        sensors = []
        for angle_deg in self.sensor_angles_deg:
            dist = self._cast_ray(self.drone_x, angle_deg, left_wall, right_wall)
            # Normalize to [0, 1]
            normalized = np.clip(dist / self.max_sensor_dist, 0.0, 1.0)
            sensors.append(normalized)

        return np.array(sensors, dtype=np.float32)

    def _cast_ray(self, drone_x, angle_deg, left_wall, right_wall):
        """
        Cast a single ray from the drone's lateral position at the given angle.
        Returns the distance to the nearest wall (left or right).

        In our simplified 1D tunnel model:
        - The ray moves laterally as it projects forward.
        - The lateral displacement at distance d is: d * tan(angle)
        - We find the d at which the ray exits through the left or right wall.
        """
        angle_rad = np.radians(angle_deg)

        # Pure forward ray (0 degrees) — use perpendicular distances
        if abs(angle_deg) < 1e-6:
            dist_left = abs(drone_x - left_wall)
            dist_right = abs(right_wall - drone_x)
            return min(dist_left, dist_right)

        tan_angle = np.tan(angle_rad)

        # Distance to left wall
        # drone_x + d * tan(angle) = left_wall  => d = (left_wall - drone_x) / tan(angle)
        d_left = (left_wall - drone_x) / tan_angle
        # Distance to right wall
        d_right = (right_wall - drone_x) / tan_angle

        # We only care about positive distances (rays going forward)
        distances = [d for d in [d_left, d_right] if d > 0]

        if not distances:
            return self.max_sensor_dist

        return min(min(distances), self.max_sensor_dist)
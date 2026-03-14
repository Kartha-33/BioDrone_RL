"""
render.py — Pygame Visualizer for BioDrone-RL (v2 — curved tunnel fix)
"""

import sys
import math
import numpy as np
import torch
import pygame

from env import BioDroneEnv
from model import FlyPolicyNetwork, get_device


# ------------------------------------------------------------------
# Visual Configuration
# ------------------------------------------------------------------

SCREEN_W        = 800
SCREEN_H        = 600
FPS             = 30
DRONE_SIZE      = 12
TUNNEL_SCALE    = 3.5
SAVE_PATH       = "weights/fly_policy.pth"
HISTORY_LEN     = 60        # How many frames of tunnel trail to show

# Colour Palette
C_BACKGROUND    = (15,  15,  25)
C_WALL_LEFT     = (0,  180, 255)
C_WALL_RIGHT    = (255, 80,  80)
C_WALL_FILL     = (30,  30,  50)
C_DRONE         = (0,  255, 150)
C_DRONE_OUTLINE = (255, 255, 255)
C_SENSOR        = (255, 220,  50)
C_TEXT          = (200, 200, 200)
C_TEXT_HIGHLIGHT= (0,  255, 150)
C_GRID          = (30,  35,  50)
C_SYNAPSE_ACTIVE= (0,  255, 150)
C_SYNAPSE_DEAD  = (40,  40,  60)


# ------------------------------------------------------------------
# Coordinate Helpers
# ------------------------------------------------------------------

def env_x_to_screen(env_x):
    return int(SCREEN_W // 2 + env_x * TUNNEL_SCALE)


# ------------------------------------------------------------------
# Load Trained Model
# ------------------------------------------------------------------

def load_model(device):
    checkpoint = torch.load(SAVE_PATH, map_location=device,
                            weights_only=False)
    hp = checkpoint['hyperparams']

    model = FlyPolicyNetwork(
        input_dim=hp['input_dim'],
        hidden_dim=hp['hidden_dim'],
        output_dim=hp['output_dim'],
        sparsity=checkpoint['sparsity']
    ).to(device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✅ Loaded weights from '{SAVE_PATH}'")
    print(f"   Trained for     : {checkpoint['episode']} episodes")
    print(f"   Best avg reward : {checkpoint['best_avg_reward']:.2f}")
    return model, checkpoint


# ------------------------------------------------------------------
# Draw Functions
# ------------------------------------------------------------------

def draw_background(screen):
    screen.fill(C_BACKGROUND)
    for x in range(0, SCREEN_W, 40):
        pygame.draw.line(screen, C_GRID, (x, 0), (x, SCREEN_H), 1)
    for y in range(0, SCREEN_H, 40):
        pygame.draw.line(screen, C_GRID, (0, y), (SCREEN_W, y), 1)


def draw_tunnel(screen, wall_history):
    """
    Render the tunnel as a scrolling ribbon.

    wall_history[-1] = most recent (at drone level, 65% down screen)
    wall_history[0]  = oldest     (near top of screen)

    Each entry is (left_wall_env_x, right_wall_env_x).
    We space them evenly from the top of the screen down to the drone.
    """
    n = len(wall_history)
    if n < 2:
        return

    drone_y    = int(SCREEN_H * 0.65)   # Drone is always here
    top_y      = 45                      # Below the HUD bar

    for i in range(n - 1):
        # Map history index to screen Y
        # i=0 (oldest) → top_y
        # i=n-1 (newest) → drone_y
        y1 = int(top_y + (i / (n - 1))       * (drone_y - top_y))
        y2 = int(top_y + ((i + 1) / (n - 1)) * (drone_y - top_y))

        lx1 = env_x_to_screen(wall_history[i][0])
        rx1 = env_x_to_screen(wall_history[i][1])
        lx2 = env_x_to_screen(wall_history[i + 1][0])
        rx2 = env_x_to_screen(wall_history[i + 1][1])

        # Tunnel fill
        poly = [(lx1, y1), (rx1, y1), (rx2, y2), (lx2, y2)]
        pygame.draw.polygon(screen, C_WALL_FILL, poly)

        # Fade walls: older = more transparent
        alpha  = int(80 + 175 * (i / (n - 1)))
        left_c  = tuple(int(c * alpha / 255) for c in C_WALL_LEFT)
        right_c = tuple(int(c * alpha / 255) for c in C_WALL_RIGHT)

        pygame.draw.line(screen, left_c,  (lx1, y1), (lx2, y2), 2)
        pygame.draw.line(screen, right_c, (rx1, y1), (rx2, y2), 2)

    # Draw current walls as full-brightness lines extending to bottom
    if wall_history:
        lx = env_x_to_screen(wall_history[-1][0])
        rx = env_x_to_screen(wall_history[-1][1])
        pygame.draw.line(screen, C_WALL_LEFT,  (lx, drone_y), (lx, SCREEN_H), 2)
        pygame.draw.line(screen, C_WALL_RIGHT, (rx, drone_y), (rx, SCREEN_H), 2)

        # Tunnel fill below drone
        below_poly = [
            (lx, drone_y), (rx, drone_y),
            (rx, SCREEN_H),(lx, SCREEN_H)
        ]
        pygame.draw.polygon(screen, C_WALL_FILL, below_poly)


def draw_drone(screen, drone_sx, drone_sy, action):
    action_colours = {
        0: (0,   200, 255),
        1: (0,   255, 150),
        2: (255, 160,  50),
    }
    colour = action_colours.get(action, C_DRONE)

    tip   = (drone_sx,              drone_sy - DRONE_SIZE)
    left  = (drone_sx - DRONE_SIZE, drone_sy + DRONE_SIZE // 2)
    right = (drone_sx + DRONE_SIZE, drone_sy + DRONE_SIZE // 2)

    pygame.draw.polygon(screen, colour,         [tip, left, right])
    pygame.draw.polygon(screen, C_DRONE_OUTLINE, [tip, left, right], 2)

    # Thruster glow
    pygame.draw.circle(screen, (255, 120, 0),
                       (drone_sx, drone_sy + DRONE_SIZE // 2), 4)


def draw_sensors(screen, drone_sx, drone_sy, obs, env):
    angles_deg = env.sensor_angles_deg
    max_dist   = env.max_sensor_dist * TUNNEL_SCALE

    for angle_deg, norm_dist in zip(angles_deg, obs):
        ray_len = float(norm_dist) * max_dist

        screen_angle_rad = math.radians(-angle_deg - 90)
        end_x = drone_sx + ray_len * math.cos(screen_angle_rad)
        end_y = drone_sy + ray_len * math.sin(screen_angle_rad)

        danger    = 1.0 - float(norm_dist)
        ray_colour= (255, int(220 * (1.0 - danger)), 50)

        pygame.draw.line(screen, ray_colour,
                         (drone_sx, drone_sy),
                         (int(end_x), int(end_y)), 1)
        pygame.draw.circle(screen, ray_colour,
                           (int(end_x), int(end_y)), 3)


def draw_connectome_panel(screen, model, font_small, font_tiny):
    px, py = 15, SCREEN_H - 165
    pw, ph = 210, 155

    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((20, 20, 35, 210))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (60, 60, 90), (px, py, pw, ph), 1)

    title = font_small.render("Connectome", True, C_TEXT_HIGHLIGHT)
    screen.blit(title, (px + 8, py + 6))

    mask1 = model.fc1.mask.cpu().numpy()   # (32, 5)
    mask2 = model.fc2.mask.cpu().numpy()   # (3,  32)

    # Layer 1 label
    lbl1 = font_tiny.render(f"L1 (5→32)", True, C_TEXT)
    screen.blit(lbl1, (px + 8, py + 22))

    # Layer 1 dots
    for row in range(mask1.shape[0]):
        for col in range(mask1.shape[1]):
            cx = px + 8 + col * 9
            cy = py + 34 + row * 3
            c  = C_SYNAPSE_ACTIVE if mask1[row, col] > 0 else C_SYNAPSE_DEAD
            pygame.draw.circle(screen, c, (cx, cy), 2)

    # Layer 2 label
    lbl2 = font_tiny.render(f"L2 (32→3)", True, C_TEXT)
    screen.blit(lbl2, (px + 65, py + 22))

    # Layer 2 dots
    for row in range(mask2.shape[0]):
        for col in range(mask2.shape[1]):
            cx = px + 65 + col * 4
            cy = py + 34 + row * 12
            c  = C_SYNAPSE_ACTIVE if mask2[row, col] > 0 else C_SYNAPSE_DEAD
            pygame.draw.circle(screen, c, (cx, cy), 2)

    active1 = int(model.fc1.mask.sum().item())
    active2 = int(model.fc2.mask.sum().item())
    stat = font_tiny.render(
        f"Active: {active1}/160   {active2}/96", True, C_TEXT
    )
    screen.blit(stat, (px + 8, py + 138))


def draw_hud(screen, font_large, font_small, font_tiny,
             episode, total_reward, step, action,
             obs, paused, best_reward):

    action_names   = ["◀ Steer Left", "▲ Go Straight", "▶ Steer Right"]
    action_colours = [(0, 200, 255), (0, 255, 150), (255, 160, 50)]

    # Top bar background
    bar = pygame.Surface((SCREEN_W, 40), pygame.SRCALPHA)
    bar.fill((10, 10, 20, 220))
    screen.blit(bar, (0, 0))

    screen.blit(font_large.render(f"Episode {episode}",     True, C_TEXT_HIGHLIGHT), (12,  5))
    screen.blit(font_large.render(f"Reward: {total_reward:.0f}", True, C_TEXT),      (210, 5))
    screen.blit(font_small.render(f"Step: {step}",          True, C_TEXT),           (430, 5))

    # Best reward (only show after first episode ends)
    best_str = f"Best: {best_reward:.0f}" if best_reward > -float('inf') else "Best: --"
    screen.blit(font_small.render(best_str, True, C_TEXT_HIGHLIGHT), (430, 22))

    # Action indicator top right
    act_surf = font_large.render(
        action_names[action], True, action_colours[action]
    )
    screen.blit(act_surf, (SCREEN_W - act_surf.get_width() - 12, 5))

    # --- Sensor panel (bottom right) ---
    labels = ["FL", "NL", "FW", "NR", "FR"]
    sx, sy = SCREEN_W - 195, SCREEN_H - 158

    panel = pygame.Surface((180, 148), pygame.SRCALPHA)
    panel.fill((20, 20, 35, 210))
    screen.blit(panel, (sx - 5, sy - 5))
    pygame.draw.rect(screen, (60, 60, 90), (sx - 5, sy - 5, 180, 148), 1)

    screen.blit(font_small.render("Sensor Readings", True, C_TEXT_HIGHLIGHT),
                (sx, sy - 2))

    for i, (lbl, val) in enumerate(zip(labels, obs)):
        by     = sy + 18 + i * 24
        bw     = int(float(val) * 130)
        danger = 1.0 - float(val)
        colour = (255, int(220 * (1.0 - danger)), 50)

        screen.blit(font_tiny.render(lbl, True, C_TEXT), (sx, by + 3))
        pygame.draw.rect(screen, (40, 40, 60),   (sx + 22, by, 130, 16))
        if bw > 0:
            pygame.draw.rect(screen, colour,     (sx + 22, by, bw,  16))
        screen.blit(font_tiny.render(f"{val:.2f}", True, C_TEXT),
                    (sx + 156, by + 3))

    # Pause overlay
    if paused:
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 130))
        screen.blit(ov, (0, 0))
        pt = font_large.render("⏸  PAUSED  —  SPACE to resume",
                               True, (255, 255, 100))
        screen.blit(pt, (SCREEN_W // 2 - pt.get_width() // 2,
                         SCREEN_H // 2 - 16))

    # Controls hint
    hint = font_tiny.render(
        "Q/ESC: Quit    SPACE: Pause    R: Restart", True, (70, 70, 90)
    )
    screen.blit(hint, (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H - 16))


# ------------------------------------------------------------------
# Main Render Loop
# ------------------------------------------------------------------

def run_visualizer():
    print("=" * 60)
    print("  BioDrone-RL | Pygame Visualizer  v2")
    print("=" * 60)

    device         = get_device()
    model, ckpt    = load_model(device)
    env            = BioDroneEnv()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(
        "BioDrone-RL | Sparse Connectome Navigation"
    )
    clock      = pygame.time.Clock()
    font_large = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    font_tiny  = pygame.font.SysFont("monospace", 11)

    print("\n🎮 Controls:  Q/ESC=Quit   SPACE=Pause   R=Restart\n")

    # --- Episode state ---
    def new_episode():
        obs, _ = env.reset()
        return obs, 0.0, 0, False, [], 1

    obs, total_reward, step, done, wall_history, action = new_episode()
    episode     = 1
    best_reward = -float('inf')
    paused      = False

    running = True
    while running:

        # ── Events ──────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, total_reward, step, done, wall_history, action = \
                        new_episode()

        if not running:
            break

        # ── Step ────────────────────────────────────────────────
        if not paused:
            if done:
                pygame.time.wait(900)
                if total_reward > best_reward:
                    best_reward = total_reward
                episode += 1
                obs, total_reward, step, done, wall_history, action = \
                    new_episode()
            else:
                # Brain decides
                with torch.no_grad():
                    obs_t  = torch.tensor(
                        obs, dtype=torch.float32
                    ).to(device)
                    action, _ = model.get_action(obs_t)

                # Record wall snapshot BEFORE stepping
                lw = env.tunnel_center_x - env.half_width
                rw = env.tunnel_center_x + env.half_width
                wall_history.append((lw, rw))
                if len(wall_history) > HISTORY_LEN:
                    wall_history.pop(0)

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                step         += 1
                done          = terminated or truncated

        # ── Draw ────────────────────────────────────────────────
        draw_background(screen)
        draw_tunnel(screen, wall_history)

        drone_sx = env_x_to_screen(env.drone_x)
        drone_sy = int(SCREEN_H * 0.65)

        draw_sensors(screen, drone_sx, drone_sy, obs, env)
        draw_drone(screen, drone_sx, drone_sy, action)
        draw_connectome_panel(screen, model, font_small, font_tiny)
        draw_hud(screen, font_large, font_small, font_tiny,
                 episode, total_reward, step, action,
                 obs, paused, best_reward)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_visualizer()
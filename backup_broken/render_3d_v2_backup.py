"""
render_3d.py — Pseudo-3D First-Person Visualizer for BioDrone-RL (v2)

Fixed perspective projection:
  - Tunnel now correctly surrounds the drone/reticle
  - Drone sits at the NEAR end of the tunnel (centre screen)
  - Tunnel extends FORWARD from the drone toward vanishing point
  - Walls fan OUT from vanishing point toward the screen edges
"""

import sys
import math
import numpy as np
import torch
import pygame

from env import BioDroneEnv
from model import FlyPolicyNetwork, get_device


# ─────────────────────────────────────────────────────────────────────────────
#  Display Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCREEN_W     = 900
SCREEN_H     = 600
FPS_DEFAULT  = 30
SAVE_PATH    = "weights/fly_policy.pth"

# Vanishing point — where the tunnel converges (upper centre)
VP_X         = SCREEN_W // 2
VP_Y         = int(SCREEN_H * 0.32)    # upper third of screen

# Drone/reticle is always at screen centre
RETICLE_X    = SCREEN_W // 2
RETICLE_Y    = int(SCREEN_H * 0.56)    # slightly below centre

# Tunnel mouth corners at the NEAR plane (surrounding the reticle)
# These are the corners of the tunnel opening right at the drone
NEAR_HW      = 310    # half-width  of tunnel mouth at near plane
NEAR_HH      = 200    # half-height of tunnel mouth at near plane

# Number of tunnel segments to draw
NUM_SEGS     = 22

# Colours
C_BG           = (8,   10,  20)
C_GRID         = (18,  22,  38)
C_WALL_L       = (0,   180, 255)    # cyan  — left wall
C_WALL_R       = (255,  60,  80)    # red   — right wall
C_WALL_TOP     = (100,  60, 200)    # purple — ceiling
C_WALL_BOT     = (40,  180,  90)    # green  — floor
C_TUNNEL_FILL  = (14,  16,  32)     # dark interior
C_TUNNEL_DARK  = (10,  12,  24)     # darker interior sides
C_RETICLE      = (0,   255, 150)
C_TEXT         = (200, 200, 200)
C_HIGHLIGHT    = (0,   255, 150)
C_SYNAPSE_ON   = (0,   255, 150)
C_SYNAPSE_OFF  = (35,  38,  55)

# Global obs for danger bar
obs_global = [1.0] * 5


# ─────────────────────────────────────────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device):
    checkpoint = torch.load(
        SAVE_PATH, map_location=device, weights_only=False
    )
    hp    = checkpoint['hyperparams']
    model = FlyPolicyNetwork(
        input_dim  = hp['input_dim'],
        hidden_dim = hp['hidden_dim'],
        output_dim = hp['output_dim'],
        sparsity   = checkpoint['sparsity']
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"✅ Weights loaded — best avg reward: "
          f"{checkpoint['best_avg_reward']:.2f}")
    return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
#  Perspective Helpers
# ─────────────────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t


def get_ring(t, curve_x=0.0, curve_y=0.0):
    """
    Compute the 4 corners of a tunnel cross-section ring.

    t = 0.0  →  near plane  (large, surrounding reticle)
    t = 1.0  →  far plane   (tiny, at vanishing point)

    curve_x : tunnel centre offset in env coords
              positive = tunnel goes right → VP shifts left
    curve_y : vertical curve (future use)

    Returns dict with corner pixel coords + metadata.
    """
    # Non-linear depth — things shrink faster near vanishing point
    depth = t ** 1.4

    # Vanishing point shifts opposite to tunnel curve
    # (if tunnel curves right, VP appears to move left)
    vp_x  = VP_X - curve_x * 22
    vp_y  = VP_Y + curve_y * 10

    # Ring centre = lerp from reticle to vanishing point
    cx = lerp(RETICLE_X, vp_x, depth)
    cy = lerp(RETICLE_Y, vp_y, depth)

    # Ring size = lerp from near-plane size to 0 at VP
    hw = NEAR_HW * (1.0 - depth)
    hh = NEAR_HH * (1.0 - depth)

    return {
        'tl': (cx - hw, cy - hh),
        'tr': (cx + hw, cy - hh),
        'bl': (cx - hw, cy + hh),
        'br': (cx + hw, cy + hh),
        'cx': cx, 'cy': cy,
        'hw': hw, 'hh': hh,
        't':  t,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: Background
# ─────────────────────────────────────────────────────────────────────────────

def draw_background(screen):
    screen.fill(C_BG)
    for x in range(0, SCREEN_W, 45):
        pygame.draw.line(screen, C_GRID, (x, 0), (x, SCREEN_H), 1)
    for y in range(0, SCREEN_H, 45):
        pygame.draw.line(screen, C_GRID, (0, y), (SCREEN_W, y), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: 3D Tunnel
# ─────────────────────────────────────────────────────────────────────────────

def draw_tunnel_3d(screen, wall_history):
    """
    Draw the pseudo-3D tunnel.

    The drone is at the NEAR end (t=0, large ring surrounding reticle).
    The tunnel extends FORWARD to the vanishing point (t=1, tiny ring).

    wall_history[-1] = current wall state (used for t=0 curve)
    wall_history[0]  = oldest state      (used for t=1 curve)
    """
    n = len(wall_history)
    if n < 2:
        # Not enough history yet — draw a straight tunnel
        wall_history_safe = [(0.0, 60.0)] * (NUM_SEGS + 1)
        n = len(wall_history_safe)
    else:
        wall_history_safe = wall_history

    segs = min(NUM_SEGS, n - 1)

    # Build rings from NEAR (t=0) to FAR (t=1)
    rings = []
    for i in range(segs + 1):
        t = i / segs

        # Map t to history index
        # t=0 → wall_history[-1] (newest = closest to drone)
        # t=1 → wall_history[0]  (oldest = furthest ahead)
        hist_idx = int((1.0 - t) * (n - 1))
        hist_idx = max(0, min(n - 1, hist_idx))

        center_x, half_w = wall_history_safe[hist_idx]

        # Normalise tunnel offset: 0 = centred, ±1 = at wall
        curve_x = center_x / max(half_w, 1.0)

        rings.append(get_ring(t, curve_x=curve_x))

    # ── Draw FAR → NEAR (painter's algorithm) ────────────────────────────
    for i in range(len(rings) - 2, -1, -1):
        r0 = rings[i]       # near ring (larger)
        r1 = rings[i + 1]   # far  ring (smaller)

        # Depth fade: near = bright, far = dark
        near_t  = i / max(len(rings) - 2, 1)   # 1=nearest seg, 0=farthest
        fade    = int(35 + 220 * near_t)

        def fc(c):
            """Fade colour by depth."""
            return tuple(max(0, min(255, int(ch * fade / 255))) for ch in c)

        # ── Fill faces ───────────────────────────────────────────────
        # Floor
        pygame.draw.polygon(screen, (12, 15, 28),
            [r1['bl'], r1['br'], r0['br'], r0['bl']])
        # Ceiling
        pygame.draw.polygon(screen, (16, 14, 30),
            [r1['tl'], r1['tr'], r0['tr'], r0['tl']])
        # Left wall
        pygame.draw.polygon(screen, (13, 15, 30),
            [r1['tl'], r1['bl'], r0['bl'], r0['tl']])
        # Right wall
        pygame.draw.polygon(screen, (13, 15, 30),
            [r1['tr'], r1['br'], r0['br'], r0['tr']])

        # ── Inner rectangle fill (tunnel interior at far ring) ───────
        pygame.draw.polygon(screen, C_TUNNEL_FILL,
            [r1['tl'], r1['tr'], r1['br'], r1['bl']])

        # ── Wall edge lines ──────────────────────────────────────────
        lw = max(1, int(3 * near_t))   # thicker near drone

        # Left wall edges
        pygame.draw.line(screen, fc(C_WALL_L),
                         _ip(r1['tl']), _ip(r0['tl']), lw)
        pygame.draw.line(screen, fc(C_WALL_L),
                         _ip(r1['bl']), _ip(r0['bl']), lw)

        # Right wall edges
        pygame.draw.line(screen, fc(C_WALL_R),
                         _ip(r1['tr']), _ip(r0['tr']), lw)
        pygame.draw.line(screen, fc(C_WALL_R),
                         _ip(r1['br']), _ip(r0['br']), lw)

        # Top (ceiling) edges
        pygame.draw.line(screen, fc(C_WALL_TOP),
                         _ip(r1['tl']), _ip(r0['tl']), lw)
        pygame.draw.line(screen, fc(C_WALL_TOP),
                         _ip(r1['tr']), _ip(r0['tr']), lw)

        # Floor edges
        pygame.draw.line(screen, fc(C_WALL_BOT),
                         _ip(r1['bl']), _ip(r0['bl']), lw)
        pygame.draw.line(screen, fc(C_WALL_BOT),
                         _ip(r1['br']), _ip(r0['br']), lw)

        # Ring outlines
        if i == 0:
            # Near plane ring — brightest outline
            pygame.draw.polygon(screen, (55, 60, 100),
                [r0['tl'], r0['tr'], r0['br'], r0['bl']], 2)

    # ── Back wall ─────────────────────────────────────────────────────────
    if len(rings) > 1:
        far = rings[-1]
        pygame.draw.polygon(screen, (22, 25, 48),
            [far['tl'], far['tr'], far['br'], far['bl']])
        pygame.draw.polygon(screen, (40, 45, 80),
            [far['tl'], far['tr'], far['br'], far['bl']], 1)
        # Vanishing point glow
        pygame.draw.circle(screen, (60, 90, 200),
                           _ip((far['cx'], far['cy'])), 5)
        pygame.draw.circle(screen, (120, 160, 255),
                           _ip((far['cx'], far['cy'])), 2)

    # ── Near plane ring  ─────────────────────────────────────────────────
    # Draw the nearest ring clearly — this is the tunnel opening
    # around the drone. Fill everything OUTSIDE this ring dark.
    if rings:
        near = rings[0]
        tl = _ip(near['tl'])
        tr = _ip(near['tr'])
        bl = _ip(near['bl'])
        br = _ip(near['br'])

        # Outside mask — darken area outside near ring
        # Left strip
        pygame.draw.rect(screen, (8, 10, 20),
                         (0, 0, tl[0], SCREEN_H))
        # Right strip
        pygame.draw.rect(screen, (8, 10, 20),
                         (tr[0], 0, SCREEN_W - tr[0], SCREEN_H))
        # Top strip
        pygame.draw.rect(screen, (8, 10, 20),
                         (tl[0], 0, tr[0] - tl[0], tl[1]))
        # Bottom strip
        pygame.draw.rect(screen, (8, 10, 20),
                         (bl[0], bl[1], br[0] - bl[0], SCREEN_H - bl[1]))

        # Near ring border
        pygame.draw.polygon(screen, (80, 90, 140),
                            [tl, tr, br, bl], 2)


def _ip(pt):
    """Convert float point tuple to int pixel coords."""
    return (int(pt[0]), int(pt[1]))


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: Sensor Rays
# ─────────────────────────────────────────────────────────────────────────────

def draw_sensor_overlay(screen, obs, env):
    """
    Sensor rays radiate FROM the reticle INTO the tunnel.
    Length proportional to distance reading.
    Colour: yellow (safe/far) → red (danger/near).
    """
    cx = RETICLE_X
    cy = RETICLE_Y
    angles_deg = env.sensor_angles_deg
    max_len    = 180

    for angle_deg, norm_dist in zip(angles_deg, obs):
        ray_len = float(norm_dist) * max_len

        # Sensors fan upward into the tunnel (toward VP)
        # angle 0 = straight ahead (up toward VP)
        screen_angle = math.radians(-angle_deg - 90)

        end_x = cx + ray_len * math.cos(screen_angle)
        end_y = cy + ray_len * math.sin(screen_angle)

        danger = 1.0 - float(norm_dist)
        col    = (255, int(220 * (1.0 - danger)), int(50 * (1.0 - danger**2)))

        pygame.draw.line(screen, col, (cx, cy),
                         (int(end_x), int(end_y)), 1)
        pygame.draw.circle(screen, col, (int(end_x), int(end_y)), 3)


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: Reticle
# ─────────────────────────────────────────────────────────────────────────────

def draw_reticle(screen, action):
    cx = RETICLE_X
    cy = RETICLE_Y

    action_colours = {
        0: (0,   200, 255),    # cyan   — steer left
        1: (0,   255, 150),    # mint   — straight
        2: (255, 160,  50),    # orange — steer right
    }
    col = action_colours.get(action, C_RETICLE)

    # Outer ring
    pygame.draw.circle(screen, col, (cx, cy), 24, 1)
    # Middle ring
    pygame.draw.circle(screen, (*col, 80), (cx, cy), 32, 1)
    # Inner dot
    pygame.draw.circle(screen, col, (cx, cy), 4)

    # Cross arms
    gap, arm = 9, 20
    pygame.draw.line(screen, col, (cx-arm, cy), (cx-gap, cy), 1)
    pygame.draw.line(screen, col, (cx+gap, cy), (cx+arm, cy), 1)
    pygame.draw.line(screen, col, (cx, cy-arm), (cx, cy-gap), 1)
    pygame.draw.line(screen, col, (cx, cy+gap), (cx, cy+arm), 1)

    # Action tick
    if action == 0:
        pygame.draw.line(screen, col,
                         (cx-arm-10, cy-7), (cx-arm-10, cy+7), 2)
    elif action == 2:
        pygame.draw.line(screen, col,
                         (cx+arm+10, cy-7), (cx+arm+10, cy+7), 2)
    else:
        pygame.draw.line(screen, col,
                         (cx-7, cy-arm-10), (cx+7, cy-arm-10), 2)

    # Draw a line from reticle toward VP (forward direction indicator)
    pygame.draw.line(screen, (*col[:3], 60),
                     (cx, cy - 28), (VP_X, VP_Y + 10), 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: Connectome Panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_connectome_panel(screen, model, font_s, font_t):
    px, py = 15, SCREEN_H - 158
    pw, ph = 210, 148

    surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
    surf.fill((12, 15, 30, 220))
    screen.blit(surf, (px, py))
    pygame.draw.rect(screen, (50, 55, 85), (px, py, pw, ph), 1)

    screen.blit(font_s.render("Connectome", True, C_HIGHLIGHT),
                (px + 8, py + 6))

    mask1 = model.fc1.mask.cpu().numpy()
    mask2 = model.fc2.mask.cpu().numpy()

    screen.blit(font_t.render("L1 (5→32)", True, C_TEXT), (px+8,  py+22))
    for row in range(mask1.shape[0]):
        for col in range(mask1.shape[1]):
            c = C_SYNAPSE_ON if mask1[row, col] > 0 else C_SYNAPSE_OFF
            pygame.draw.circle(screen, c,
                               (px + 8 + col * 9, py + 34 + row * 3), 2)

    screen.blit(font_t.render("L2 (32→3)", True, C_TEXT), (px+65, py+22))
    for row in range(mask2.shape[0]):
        for col in range(mask2.shape[1]):
            c = C_SYNAPSE_ON if mask2[row, col] > 0 else C_SYNAPSE_OFF
            pygame.draw.circle(screen, c,
                               (px + 65 + col * 4, py + 34 + row * 12), 2)

    a1 = int(model.fc1.mask.sum().item())
    a2 = int(model.fc2.mask.sum().item())
    screen.blit(font_t.render(f"Active: {a1}/160   {a2}/96", True, C_TEXT),
                (px + 8, py + 132))


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: Sensor Bars
# ─────────────────────────────────────────────────────────────────────────────

def draw_sensor_bars(screen, obs, font_s, font_t):
    labels = ["FL", "NL", "FW", "NR", "FR"]
    sx, sy = SCREEN_W - 195, SCREEN_H - 158

    surf = pygame.Surface((180, 148), pygame.SRCALPHA)
    surf.fill((12, 15, 30, 220))
    screen.blit(surf, (sx - 5, sy - 5))
    pygame.draw.rect(screen, (50, 55, 85), (sx - 5, sy - 5, 180, 148), 1)

    screen.blit(font_s.render("Sensor Readings", True, C_HIGHLIGHT),
                (sx, sy - 2))

    for i, (lbl, val) in enumerate(zip(labels, obs)):
        by     = sy + 18 + i * 24
        bw     = int(float(val) * 130)
        danger = 1.0 - float(val)
        colour = (255, int(220 * (1.0 - danger)), 50)

        screen.blit(font_t.render(lbl, True, C_TEXT), (sx, by + 3))
        pygame.draw.rect(screen, (35, 38, 58), (sx + 22, by, 130, 16))
        if bw > 0:
            pygame.draw.rect(screen, colour, (sx + 22, by, bw, 16))
        screen.blit(font_t.render(f"{val:.2f}", True, C_TEXT),
                    (sx + 156, by + 3))


# ─────────────────────────────────────────────────────────────────────────────
#  Draw: HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(screen, font_l, font_s, font_t,
             episode, total_reward, step, action,
             paused, best_reward, fps_actual):

    action_names   = ["◀ LEFT", "▲ STRAIGHT", "▶ RIGHT"]
    action_colours = [(0, 200, 255), (0, 255, 150), (255, 160, 50)]

    # Top bar
    bar = pygame.Surface((SCREEN_W, 44), pygame.SRCALPHA)
    bar.fill((8, 10, 22, 235))
    screen.blit(bar, (0, 0))
    pygame.draw.line(screen, (40, 45, 75), (0, 44), (SCREEN_W, 44), 1)

    screen.blit(font_l.render(f"Episode {episode}",
                True, C_HIGHLIGHT), (12, 6))
    screen.blit(font_l.render(f"Reward: {total_reward:.0f}",
                True, C_TEXT),      (210, 6))
    screen.blit(font_s.render(f"Step: {step}",
                True, C_TEXT),      (430, 8))

    best_str = f"Best: {best_reward:.0f}" \
               if best_reward > -1e9 else "Best: --"
    screen.blit(font_s.render(best_str, True, C_HIGHLIGHT), (430, 26))
    screen.blit(font_t.render(f"FPS: {fps_actual:.0f}",
                True, (70, 80, 110)), (530, 28))

    # Action indicator — top right
    act = font_l.render(action_names[action], True, action_colours[action])
    screen.blit(act, (SCREEN_W - act.get_width() - 14, 6))

    # ── Danger bar ────────────────────────────────────────────────────────
    min_sensor = float(min(obs_global))
    danger     = max(0.0, 1.0 - min_sensor)
    bar_w      = int(SCREEN_W * danger)
    if bar_w > 0:
        dc = (int(255 * danger), int(80 * (1.0 - danger)), 30)
        ds = pygame.Surface((bar_w, 3), pygame.SRCALPHA)
        ds.fill((*dc, 200))
        screen.blit(ds, (0, 41))

    # ── Pause overlay ─────────────────────────────────────────────────────
    if paused:
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 145))
        screen.blit(ov, (0, 0))
        pt = font_l.render("⏸  PAUSED  —  SPACE to resume",
                           True, (255, 255, 100))
        screen.blit(pt, (SCREEN_W//2 - pt.get_width()//2,
                         SCREEN_H//2 - 16))

    # Controls hint
    hint = font_t.render(
        "Q/ESC: Quit    SPACE: Pause    R: Restart    +/-: Speed",
        True, (55, 60, 82)
    )
    screen.blit(hint, (SCREEN_W//2 - hint.get_width()//2, SCREEN_H - 16))


# ─────────────────────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_visualizer():
    global obs_global

    print("=" * 60)
    print("  BioDrone-RL | Pseudo-3D First-Person Visualizer  v2")
    print("=" * 60)

    device      = get_device()
    model, ckpt = load_model(device)
    env         = BioDroneEnv()

    pygame.init()
    screen  = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(
        "BioDrone-RL | First-Person Sparse Connectome Navigation"
    )
    clock      = pygame.time.Clock()
    font_large = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    font_tiny  = pygame.font.SysFont("monospace", 11)

    print("\n🎮 Controls:")
    print("   Q / ESC : Quit")
    print("   SPACE   : Pause / Resume")
    print("   R       : Restart episode")
    print("   + / -   : Speed up / slow down\n")

    # ── Episode state ──────────────────────────────────────────────────────
    def new_episode():
        o, _ = env.reset()
        return o, 0.0, 0, False, [], 1

    obs, total_reward, step, done, wall_history, action = new_episode()
    obs_global  = list(obs)
    episode     = 1
    best_reward = -float('inf')
    paused      = False
    fps_target  = FPS_DEFAULT
    fps_actual  = float(FPS_DEFAULT)

    running = True
    while running:

        # ── Events ────────────────────────────────────────────────────
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
                    obs_global = list(obs)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    fps_target = min(120, fps_target + 10)
                    print(f"   Speed: {fps_target} FPS")
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps_target = max(5, fps_target - 10)
                    print(f"   Speed: {fps_target} FPS")

        if not running:
            break

        # ── Step ──────────────────────────────────────────────────────
        if not paused:
            if done:
                pygame.time.wait(700)
                if total_reward > best_reward:
                    best_reward = total_reward
                episode += 1
                obs, total_reward, step, done, wall_history, action = \
                    new_episode()
                obs_global = list(obs)
            else:
                with torch.no_grad():
                    obs_t     = torch.tensor(
                        obs, dtype=torch.float32
                    ).to(device)
                    action, _ = model.get_action(obs_t)

                # Store wall state for history
                wall_history.append((
                    env.tunnel_center_x,
                    env.half_width
                ))
                if len(wall_history) > NUM_SEGS + 8:
                    wall_history.pop(0)

                obs, reward, terminated, truncated, _ = env.step(action)
                obs_global   = list(obs)
                total_reward += reward
                step         += 1
                done          = terminated or truncated

        # ── Draw ──────────────────────────────────────────────────────
        draw_background(screen)
        draw_tunnel_3d(screen, wall_history)
        draw_sensor_overlay(screen, obs, env)
        draw_reticle(screen, action)
        draw_connectome_panel(screen, model, font_small, font_tiny)
        draw_sensor_bars(screen, obs, font_small, font_tiny)
        draw_hud(
            screen, font_large, font_small, font_tiny,
            episode, total_reward, step, action,
            paused, best_reward, fps_actual
        )

        pygame.display.flip()
        dt         = clock.tick(fps_target)
        fps_actual = 1000.0 / max(dt, 1)

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_visualizer()
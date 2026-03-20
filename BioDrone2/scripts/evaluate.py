import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import os
import sys
import argparse

# Path Hack
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from envs.drone_3d_env import Drone3DEnv
from models.bio_policy import ActorCritic

def visualize(model_path, output_file):
    print(f"Loading model from {model_path}...")
    
    # Init Env & Model
    env = Drone3DEnv()
    device = torch.device('cpu')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = ActorCritic(state_dim, action_dim, hidden_dim=128).to(device)
    
    # Load Model (Graceful Fail)
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}")
        print("Using random weights.")

    # Reset
    state, _ = env.reset()
    frames_data = []
    
    print("Simulating episode...")
    # Run to completion or 1000 steps
    for t in range(1000):
        # Infer
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, _, _ = policy.get_action(state_t)
        action_np = action.squeeze(0).detach().numpy()
        
        # Step
        next_state, reward, done, trunc, _ = env.step(action_np)
        
        # Store Data needed for plotting
        frames_data.append({
            'time': t,
            'pos': env.state[0:3].copy(),
            'obstacles': [o.copy() for o in env.obstacles] # Copy current obstacles
        })
        
        state = next_state
        if done or trunc:
            print(f"Finished at step {t}")
            break

    # Setup Plot (3 Subplots)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Top-Down (X vs Y)
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_title("Top-Down View (Navigation)")
    ax_top.set_xlabel("Distance (X)")
    ax_top.set_ylabel("Lateral (Y)")
    ax_top.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Side View (X vs Z)
    ax_side = fig.add_subplot(gs[1, 0])
    ax_side.set_title("Side View (Altitude)")
    ax_side.set_xlabel("Distance (X)")
    ax_side.set_ylabel("Height (Z)")
    ax_side.set_ylim(0, 4)
    ax_side.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Rear View (Y vs Z)
    ax_rear = fig.add_subplot(gs[1, 1])
    ax_rear.set_title("Rear View (Pilot)")
    ax_rear.set_xlabel("Lateral (Y)")
    ax_rear.set_ylabel("Height (Z)")
    ax_rear.set_xlim(-4, 4)
    ax_rear.set_ylim(0, 4)
    ax_rear.grid(True, linestyle='--', alpha=0.5)
    
    # Static Elements
    wall1_top, = ax_top.plot([], [], 'k-', linewidth=3)
    wall2_top, = ax_top.plot([], [], 'k-', linewidth=3)
    floor_side, = ax_side.plot([], [], 'k-', linewidth=2)
    ceil_side, = ax_side.plot([], [], 'k-', linewidth=2)
    
    # Drone & Trails
    drone_top, = ax_top.plot([], [], 'ro', markersize=8, label='Drone')
    trail_top, = ax_top.plot([], [], 'r:', linewidth=1)
    
    drone_side, = ax_side.plot([], [], 'bo', markersize=8)
    trail_side, = ax_side.plot([], [], 'b:', linewidth=1)
    
    drone_rear, = ax_rear.plot([], [], 'go', markersize=10)
    info_txt = ax_top.text(0.02, 0.9, "", transform=ax_top.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Obstacle patches containers
    obs_patches_top = []
    obs_patches_side = []
    
    def init():
        return drone_top, trail_top, drone_side, trail_side, drone_rear, info_txt

    def update(frame_idx):
        data = frames_data[frame_idx]
        pos = data['pos']
        obstacles = data['obstacles']
        
        # Window logic
        window_start = max(-1, pos[0] - 5)
        window_end = window_start + 20
        
        ax_top.set_xlim(window_start, window_end)
        ax_top.set_ylim(-4, 4)
        ax_side.set_xlim(window_start, window_end)
        
        # Walls/Floor update
        wall1_top.set_data([window_start, window_end], [-3, -3])
        wall2_top.set_data([window_start, window_end], [3, 3])
        floor_side.set_data([window_start, window_end], [0.1, 0.1])
        ceil_side.set_data([window_start, window_end], [3, 3])
        
        # Drone
        drone_top.set_data([pos[0]], [pos[1]])
        drone_side.set_data([pos[0]], [pos[2]])
        drone_rear.set_data([pos[1]], [pos[2]])
        
        # Trails
        past = frames_data[max(0, frame_idx-50):frame_idx+1] # Short trail for perf
        tx = [p['pos'][0] for p in past]
        ty = [p['pos'][1] for p in past]
        tz = [p['pos'][2] for p in past]
        trail_top.set_data(tx, ty)
        trail_side.set_data(tx, tz)
        
        info_txt.set_text(f"T={data['time']} | X={pos[0]:.1f}m | H={pos[2]:.1f}")
        
        # Draw Obstacles (Red Boxes)
        # Clear old patches (inefficient but simple)
        [p.remove() for p in obs_patches_top]
        [p.remove() for p in obs_patches_side]
        obs_patches_top.clear()
        obs_patches_side.clear()
        
        for box in obstacles:
            # Box: [xmin, xmax, ymin, ymax, zmin, zmax]
            if box[1] < window_start or box[0] > window_end: continue
            
            # Top View Rect (x, y, w, h)
            rect_top = patches.Rectangle((box[0], box[2]), box[1]-box[0], box[3]-box[2], 
                                         linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
            ax_top.add_patch(rect_top)
            obs_patches_top.append(rect_top)
            
            # Side View Rect (x, z, w, h)
            rect_side = patches.Rectangle((box[0], box[4]), box[1]-box[0], box[5]-box[4], 
                                          linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
            ax_side.add_patch(rect_side)
            obs_patches_side.append(rect_side)
            
        return drone_top, trail_top, drone_side, trail_side, drone_rear, info_txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames_data), init_func=init, interval=30)
    print(f"Rendering dashboard to {output_file}...")
    ani.save(output_file, writer='ffmpeg', fps=30)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weights/champion_obstacles_final.pt', help='Path to weights')
    parser.add_argument('--output_file', type=str, default='renders/phase_f_obstacles.mp4', help='Output video')
    args = parser.parse_args()
    visualize(args.model_path, args.output_file)
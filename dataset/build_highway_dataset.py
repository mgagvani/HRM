import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import time
import math

# Enable interactive mode for live plotting
plt.ion()

# --- Configuration tweaks (easy to change) ---
GRID_SIZE = [[-25, 75], [-30, 30]]  # longitudinal, lateral span (meters)
GRID_STEP = [6, 3]                  # cell size (meters); try [12,3] or [2,2]
SHOW_DIAGNOSTICS_EVERY = 5          # steps
AUTO_SCALE_FEATURES = True          # autoscale non-binary features each frame
PRESENCE_FIXED_SCALE = (0, 1)       # keep presence 0..1

def create_env() -> gym.Env:
    """Create and return a configured highway-env environment."""
    env = gym.make(
        'intersection-v0',
        render_mode='rgb_array',
        config={
            "observation": {
                "type": "OccupancyGrid",
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features": ["presence"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "grid_size": GRID_SIZE,
                "grid_step": GRID_STEP,
                "absolute": False,
            },
            "lanes_count": 7,
            "vehicles_count": 50,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        },
    )
    return env

# ---- Helper functions ----

def _split_planes(obs: np.ndarray, feature_names):
    """Return planes array shaped (F,H,W) regardless of original layout.
    highway-env may deliver (F,H,W) (current default) or (H,W,F) in some forks.
    """
    if obs.ndim != 3:
        raise ValueError(f"Unexpected obs ndim {obs.ndim}")
    F = len(feature_names)
    if obs.shape[0] == F:  # (F,H,W)
        return obs
    if obs.shape[-1] == F:  # (H,W,F) -> transpose
        return np.moveaxis(obs, -1, 0)
    raise ValueError(f"Cannot infer feature axis from shape {obs.shape} vs F={F}")

def occupancy_stats(obs: np.ndarray, feature_names):
    planes = _split_planes(obs, feature_names)
    F, H, W = planes.shape
    total_cells = H * W
    stats = []
    for i, name in enumerate(feature_names):
        plane = planes[i]
        nz = int(np.count_nonzero(plane))
        maxv = float(plane.max()) if nz else 0.0
        minv = float(plane.min()) if nz else 0.0
        stats.append((name, nz, f"{nz/total_cells:.2%}", minv, maxv))
    return stats

def print_stats(step, obs, feature_names):
    stats = occupancy_stats(obs, feature_names)
    print(f"[Step {step}] OccupancyGrid stats:")
    for name, nz, pct, mn, mx in stats:
        print(f"  {name:<8} nz={nz:4d} ({pct})  min={mn:+.3f} max={mx:+.3f}")

from copy import deepcopy

def simulate_future_waypoints(env, horizon=10, spacing=1):
    sim_env = deepcopy(env)
    wps = []
    for k in range(horizon):
        # Choose expert action (keep_lane / IDM default):
        # TODO: implement IDM policy here, probably
        action = sim_env.action_space.sample()  # replace with heuristic
        sim_env.step(action)
        x,y = sim_env.vehicle.position
        wps.append((x,y))
    # Transform to ego frame of ORIGINAL env (not sim_env)
    ego = env.vehicle
    theta = ego.heading
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    ego_x, ego_y = ego.position
    out=[]
    for (xw,yw) in wps:
        dx, dy = xw-ego_x, yw-ego_y
        x_ego = dx*cos_t + dy*sin_t
        y_ego = -dx*sin_t + dy*cos_t
        out.append((x_ego,y_ego))
    return np.array(out, dtype=np.float32)

# ---- Main execution ----

def main():
    # Create subplots for visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Highway Environment - Live Visualization')

    # Initialize plots
    env = create_env()
    env.reset()
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)

    # Normalize to (F,H,W) for plotting
    feature_names = ["presence"]
    planes = _split_planes(obs, feature_names)
    F, H, W = planes.shape

    # Setup image plots
    im_render = axes[0, 0].imshow(env.render())
    axes[0, 0].set_title('Highway Render')
    axes[0, 0].axis('off')

    # Setup occupancy grid feature plots
    im_features = []
    for i, feature_name in enumerate(feature_names):
        row = (i + 1) // 4
        col = (i + 1) % 4
        if feature_name == "presence":
            cmap = 'gray_r'
            vmin, vmax = PRESENCE_FIXED_SCALE
        else:
            cmap = 'viridis'
            vmin, vmax = -1, 1
        im_feat = axes[row, col].imshow(planes[i], cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        axes[row, col].set_title(feature_name)
        axes[row, col].axis('off')
        im_features.append(im_feat)
        plt.colorbar(im_feat, ax=axes[row, col], shrink=0.8)

    plt.tight_layout()

    # Initial diagnostics
    print_stats(0, obs, feature_names)
    print("Tip: If nz=0 for presence, expand grid_size or increase cell size (larger grid_step), or confirm vehicles inside bounds.\n")

    # Run simulation with live updates
    for step in range(1, 201):
        if step % 10 == 0:
            action = env.action_space.sample()
        else:
            action = env.unwrapped.action_type.actions_indexes["IDLE"]

        obs, reward, done, truncated, info = env.step(action)
        planes = _split_planes(obs, feature_names)

        if step == 1:
            print("Observation raw shape:", obs.shape, "-> planes shape:", planes.shape, "(F,H,W)")

        im_render.set_array(env.render())

        for i, im_feat in enumerate(im_features):
            plane = planes[i]
            if i == 0:  # presence
                im_feat.set_array(plane)
                im_feat.set_clim(*PRESENCE_FIXED_SCALE)
            else:
                im_feat.set_array(plane)
                if AUTO_SCALE_FEATURES:
                    if np.any(plane):
                        pmin, pmax = plane.min(), plane.max()
                        if pmin == pmax:
                            pmin -= 1e-3; pmax += 1e-3
                        im_feat.set_clim(pmin, pmax)
                    else:
                        im_feat.set_clim(-1, 1)

        fig.suptitle(
            f'Highway Env | Step {step} Reward {reward:.3f} Done {done} | grid_step={GRID_STEP} grid_size={GRID_SIZE}'
        )

        if step % SHOW_DIAGNOSTICS_EVERY == 0:
            print_stats(step, obs, feature_names)

        plt.draw(); plt.pause(0.05)

        if done or truncated:
            print(f"Episode ended at step {step}; resetting.")
            obs, _ = env.reset()

    plt.ioff(); plt.show(); env.close()

if __name__ == "__main__":
    main()

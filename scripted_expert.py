from env.rocket_env import SimRocketEnv
import numpy as np
import json
import random

OUTPUT_FILE = "expert_data.json"
MAX_EPISODES = 500

expert_data = []
env = SimRocketEnv(interactive=False)

for ep in range(MAX_EPISODES):
    # randomimze landing target in ±10 m
    tx = random.uniform(-10.0, 10.0)
    ty = random.uniform(-10.0, 10.0)
    print(f"[DEBUG] Episode {ep+1}: target = ({tx:.2f}, {ty:.2f})")

    # temp clamp parameters for tilt, eh
    max_tilt = np.deg2rad(10)
    k_pz, k_dz = 0.8, 0.3
    k_pa = 0.5
    kp_att = 2.0

    #reset and run expert till last episode
    state, _ = env.reset(mode="launch")
    done = False
    while not done:
        # vertical thrust
        z_err    = env.pos_n[2]
        zdot_err = env.vel_n[2]
        thrust_fb = k_pz * (-z_err) + k_dz * (-zdot_err)
        thrust    = np.clip(thrust_fb, 0.0, 1.0)

        # horizontal bearing calc
        dx = tx - env.pos_n[0]
        dy = ty - env.pos_n[1]
        desired_bearing = np.arctan2(dy, dx)
        current_yaw     = np.deg2rad(env.yaw_deg)
        yaw_err = ((desired_bearing - current_yaw + np.pi) % (2*np.pi)) - np.pi
        alpha = np.clip(k_pa * yaw_err, -max_tilt, max_tilt)
        beta  = 0.0

        # stabilization(roll & pitch)
        roll_rad  = np.deg2rad(env.roll_deg)
        pitch_rad = np.deg2rad(env.pitch_deg)
        att_roll  = np.clip(-kp_att * roll_rad,  -1.0, 1.0)
        att_pitch = np.clip(-kp_att * pitch_rad, -1.0, 1.0)

        # final assembly of controls 
        u = np.array([thrust, alpha, beta, att_roll, att_pitch])

        # step and record data
        # obs = [qw, qx, qy, qz, x, y, z, vx, vy, vz, roll_deg, pitch_deg, yaw_deg]
        obs, _, done, _, _ = env.step(u)
        expert_data.append({
            "obs":    obs.tolist(),    # 16 floats
            "target": [tx, ty],        # 2 floats
            "acts":   u.tolist()       # 5 floats
        })

    if (ep+1) % 50 == 0:
        print(f"Collected {ep+1}/{MAX_EPISODES} episodes")

with open(OUTPUT_FILE, "w") as f:
    json.dump(expert_data, f, indent=2)
print(f"Saved {len(expert_data)} samples to {OUTPUT_FILE}")

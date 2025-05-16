#!/usr/bin/env python3
import time
import argparse
import numpy as np
from env.rocket_env import SimRocketEnv
from nnpolicy import NNPolicy


def main():
    
    # added user input for target coordinates
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx", type=float, default=10.0, help="Landing target X coordinate")
    parser.add_argument("--ty", type=float, default=5.0,  help="Landing target Y coordinate")
    args = parser.parse_args()

    target = (args.tx, args.ty)
    
    # 1) Initialize environment and policy
    env = SimRocketEnv(interactive=True)
    #target = (-10.0, -5.0)
    policy = NNPolicy(
        model_file="torch_nn_mpc_scripted.pth",
        target_xy=target
    )

    # 2) Reset environment
    state, _ = env.reset(mode="launch")
    done = False

    while not done:
        # Query policy for next action
        action = policy.next(state)

        # Step simulation
        state, reward, done, _, _ = env.step(action)

        # Print state and action
        env.print_state()
        print(f"Action: thrust={action[0]:.2f}, alpha={action[1]:.2f}, beta={action[2]:.2f}, att_x={action[3]:.2f}, att_y={action[4]:.2f}")

        # Cap frame rate
        time.sleep(env.dt_sec)

    print("Episode finished with reward: ", reward)
    env.close()


if __name__ == '__main__':
    main()



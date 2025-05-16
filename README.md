# PyRocketCraft AI Landing Demo

This project implements a rocket‐landing simulator controlled by a Behavioral Cloning neural network.

# 1. Install dependencies

pip install -r requirements.txt

# 2. Generate expert data

python scripted_expert.py

# 3. Train the NN policy

python expert_train.py

# 4. Run landing demo with a given target

python run_rocket.py --tx 10.0 --ty 5.0

.
├─ rocket_env.py          # Gym‐style env wrapping PyBullet
├─ scripted_expert.py     # collects (obs, target acts) → expert_data.json
├─ expert_train.py        # trains BC policy → bc_policy.pth
├─ nnpolicynetwork.py     # 3‐layer MLP definition
├─ nnpolicy.py            # loads .pth, does inference on (obs, target)
├─ run_rocket.py          # reset+loop: policy.next() → env.step()
├─ requirements.txt       # Python deps
└─ modelrocket.urdf            # rocket model

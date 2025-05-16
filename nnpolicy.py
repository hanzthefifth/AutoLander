import numpy as np, torch
from nnpolicynetwork import NNPolicyNetwork
from env.rocket_env import SimRocketEnv

class NNPolicy:
    def __init__(self, model_file:str, target_xy:tuple):
        # derive obs dim
        env = SimRocketEnv(interactive=False)
        obs_dim = env.observation_space.shape[0]
        in_size = obs_dim + 2  # obs + target

        self.target = np.array(target_xy, dtype=np.float32)
        self.device = torch.device("cpu")
        self.model  = NNPolicyNetwork(in_size, 5)
        self.model.load_state_dict(torch.load(model_file, map_location="cpu"))
        self.model.eval()

    def next(self, obs:np.ndarray):
        inp = np.concatenate([obs, self.target])
        with torch.no_grad():
            t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            a = self.model(t).cpu().numpy().ravel()
        return a

    def get_name(self) -> str:
        return "NN"
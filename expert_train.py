import json
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from nnpolicynetwork import NNPolicyNetwork

def train_and_evaluate():
    INPUT_FILE = "expert_data.json"
    OUTPUT_FILE = "torch_nn_mpc_scripted.pth"
    BATCH_SIZE = 128
    LR = 5e-4
    EPOCHS = 10
    VAL_SPLIT = 0.05
    SEED = 42

    print("Loading expert data from:", INPUT_FILE)
    with open(INPUT_FILE) as f:
        data = json.load(f)

    np.random.seed(SEED)
    obs = np.array([d["obs"] + d["target"] for d in data], dtype=np.float32)
    acts = np.array([d["acts"] for d in data], dtype=np.float32)

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(acts)
    dataset = TensorDataset(obs_t, act_t)

    n_val = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    input_size  = obs.shape[1]  # 18 dims
    output_size = acts.shape[1] # 5 dims
    model = NNPolicyNetwork(input_size, output_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    start = time.time()
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()
        val_loss = sum(loss_fn(model(xb.to(device)), yb.to(device)).item()
                       for xb,yb in val_loader)
        print(f"Epoch {epoch:02d} | Train: {total_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), OUTPUT_FILE)
    print(f"Training done in {time.time()-start:.1f}s. Model saved to {OUTPUT_FILE}.")

if __name__ == '__main__':
    train_and_evaluate()



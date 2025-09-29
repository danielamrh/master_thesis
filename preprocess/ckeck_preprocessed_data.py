import torch

path = "/home/danielamrhein/master_thesis/preprocess/preprocessed_data/processed_04_session1_0.pkl_ekf.pt"
data = torch.load(path, map_location=torch.device("cpu"))

print("Keys in preprocessed data:", data.keys())
print("acc_world shape:", data["acc_world"].shape)  # (T, S
print("vel shape:", data["vel"].shape)              # (T, S, 3)
print("pos shape:", data["pos"].shape)              # (T, S,
print("quat shape:", data["quat"].shape)            # (T, S, 4)
print("vuwb shape:", data["vuwb"].shape)            # (T, P)

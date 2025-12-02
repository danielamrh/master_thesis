import torch

# Load dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

# Select one sequence
acc = data["acc"][0]  # (frames, 6, 3)

frames_to_check = [0, 50, 100, 500, 1000]  # some sample frames

for t in frames_to_check:
    print(f"\n=== Frame {t} ===")
    for s in range(6):
        print(f"Sensor {s}: {acc[t, s].tolist()}")

import torch

# Load your dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

# Pick one sequence (first one)
vuwb = data["vuwb"][0]  # shape (frames, 6, 6)

print("vuwb shape:", vuwb.shape)  # e.g., (17055, 6, 6)

# Look at the very first frame
frame0_vuwb = vuwb[0]  # shape (6, 6)

print("\nFirst frame virtual UWB distance matrix:")
print(frame0_vuwb)

# Print pairwise distances sensor by sensor
print("\nPairwise distances at first frame:")
for i in range(frame0_vuwb.shape[0]):
    for j in range(i + 1, frame0_vuwb.shape[1]):  # upper triangle only
        print(f"  Distance Sensor {i} ↔ Sensor {j}: {frame0_vuwb[i,j].item():.4f}")

# Flatten into vector form if you like (all 36 entries)
flattened = frame0_vuwb.flatten()
print("\nFlattened distance matrix (all 36 values):")
print(flattened.tolist())

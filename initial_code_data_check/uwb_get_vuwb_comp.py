import torch
import matplotlib.pyplot as plt

# Load dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

# Pick one sequence (first one)
vuwb = data["vuwb"][0]     # shape (frames, 6, 6)
uwb_gt = data["uwb_gt"][0] # shape (frames, 6, 6)

print("vuwb shape:", vuwb.shape)
print("uwb_gt shape:", uwb_gt.shape)

# Inspect first frame
frame0_vuwb = vuwb[0]
frame0_gt   = uwb_gt[0]

print("\nFirst frame virtual UWB distance matrix:")
print(frame0_vuwb)

print("\nFirst frame ground-truth UWB distance matrix:")
print(frame0_gt)

# Compare one pair of sensors (e.g., Sensor 0 ↔ Sensor 1)
sensor_i, sensor_j = 0, 1

vuwb_dist = vuwb[:, sensor_i, sensor_j].numpy()
gt_dist   = uwb_gt[:, sensor_i, sensor_j].numpy()

# Plot comparison
plt.figure(figsize=(12, 4))
plt.plot(vuwb_dist, label="vuwb (virtual)")
plt.plot(gt_dist, label="uwb_gt (ground truth)")
plt.title(f"Distance over time (Sensor {sensor_i} ↔ Sensor {sensor_j})")
plt.xlabel("Frame")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True)
plt.show()

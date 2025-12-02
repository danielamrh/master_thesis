import torch

# Load your dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

# Pick one sequence (e.g., first one)
acc = data["acc"][0]  # shape (frames, 6, 3)

print("acc shape:", acc.shape)  # should print (17055, 6, 3)

# Look at the very first frame
frame0 = acc[0]  # shape (6, 3)

print("\nFirst frame acceleration values (per sensor):")
for sensor_id in range(frame0.shape[0]):
    x, y, z = frame0[sensor_id].tolist()
    print(f"  Sensor {sensor_id}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

# If you want the flattened version (all sensors in one row)
flattened = frame0.flatten()
print("\nFlattened (x,y,z for all 6 sensors):")
print(flattened.tolist())

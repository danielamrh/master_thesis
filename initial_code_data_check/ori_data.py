import torch

# Load your dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

# Pick one sequence (e.g., first one)
ori = data["ori"][0]  # shape (frames, 6, 3, 3)

print("ori shape:", ori.shape)  # e.g., (17055, 6, 3, 3)

# Look at the very first frame
frame0_ori = ori[0]  # shape (6, 3, 3)

print("\nFirst frame orientation matrices (per sensor):")
for sensor_id in range(frame0_ori.shape[0]):
    R = frame0_ori[sensor_id]  # 3x3 rotation matrix
    print(f"\n  Sensor {sensor_id} rotation matrix:")
    print(R)

# If you want to flatten one sensor’s matrix (row by row)
sensor0_R = frame0_ori[0].flatten()
print("\nFlattened rotation matrix for Sensor 0:")
print(sensor0_R.tolist())

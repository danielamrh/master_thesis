import torch

# Load dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device('cpu'))

acc = data["acc"][0]   # (frames, 6, 3)
ori = data["ori"][0]   # (frames, 6, 3, 3)

print("acc shape:", acc.shape)
print("ori shape:", ori.shape)

# === Step 1: Transform acc using ori ===
# body → world via matrix multiplication
acc_world_check = torch.einsum("tnij,tnj->tni", ori, acc)

# === Step 2: Compare ===
diff = acc_world_check - acc
max_diff = diff.abs().max().item()
mean_diff = diff.abs().mean().item()

print("\n=== Sanity check: acc vs ori @ acc ===")
print(f"Max abs difference:  {max_diff:.6e}")
print(f"Mean abs difference: {mean_diff:.6e}")

tol = 1e-6
if max_diff < tol:
    print("Accelerations are already in world frame.")
else:
    print("Warning: Accelerations may still be in body frame!")

# === Step 3: Show some example frames ===
frames_to_check = [0, 100, 1000, 5000, 10000]

for t in frames_to_check:
    if t >= acc.shape[0]:
        continue
    print(f"\n=== Frame {t} ===")
    for s in range(acc.shape[1]):  # 6 sensors
        body_vec = acc[t, s]
        ori_mat = ori[t, s]
        world_vec = acc_world_check[t, s]

        print(f"Sensor {s}:")
        print(f"  body        = {body_vec.tolist()}")
        print(f"  ori_mat =\n{ori_mat}")
        print(f"  ori @ body  = {world_vec.tolist()}")
        print(f"  stored acc  = {acc[t, s].tolist()}")

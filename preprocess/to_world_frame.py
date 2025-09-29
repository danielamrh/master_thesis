import os
import torch
from scipy.spatial.transform import Rotation as R

def to_world_frame(acc_body: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tsij,tsj->tsi", ori, acc_body)

def ori_to_quat(ori: torch.Tensor) -> torch.Tensor:
    T, S, _, _ = ori.shape
    ori_np = ori.reshape(-1, 3, 3).numpy()
    quats = R.from_matrix(ori_np).as_quat()
    return torch.from_numpy(quats).view(T, S, 4)

def integrate_acc(acc_world: torch.Tensor, dt: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    vel = torch.cumsum(acc_world * dt, dim=0)
    pos = torch.cumsum(vel * dt, dim=0)
    return vel, pos

# Preprocessing
pkl_path = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/train.pt"
data = torch.load(pkl_path, map_location=torch.device("cpu"))

# Save folders
save_dir_all = "/home/danielamrhein/master_thesis/preprocess/preprocessed_data"
save_dir_arm = "/home/danielamrhein/master_thesis/preprocess/preprocessed_data_arm_trunk"
os.makedirs(save_dir_all, exist_ok=True)
os.makedirs(save_dir_arm, exist_ok=True)

# Arm + trunk sensor indices (example: trunk=5, left wrist=0, right wrist=1)
ARM_TRUNK_SENSORS = [0, 5]

for seq_idx, fname in enumerate(data["fnames"]):
    print(f"Processing sequence {seq_idx+1}/{len(data['fnames'])}: {fname}")

    acc = data["acc"][seq_idx]   # (frames, S, 3)
    ori = data["ori"][seq_idx]   # (frames, S, 3, 3)
    vuwb = data["vuwb"][seq_idx] # (frames, P)

    # --- EKF-ready all sensors ---
    acc_world = to_world_frame(acc, ori)
    vel, pos = integrate_acc(acc_world, dt=0.01)
    quats = ori_to_quat(ori).to(dtype=torch.float32)

    ekf_data_all = {
        "acc_world": acc_world,
        "vel": vel,
        "pos": pos,
        "quat": quats,
        "vuwb": vuwb
    }

    base_name = os.path.basename(fname)
    save_path_all = os.path.join(save_dir_all, f"{base_name}_ekf.pt")
    torch.save(ekf_data_all, save_path_all)
    print(f"Saved all-sensor EKF data to {save_path_all}")

    # --- EKF-ready arm+trunk only ---
    acc_world_arm = acc_world[:, ARM_TRUNK_SENSORS]
    vel_arm = vel[:, ARM_TRUNK_SENSORS]
    pos_arm = pos[:, ARM_TRUNK_SENSORS]
    quats_arm = quats[:, ARM_TRUNK_SENSORS]

    # Optional: UWB distances for arm+trunk sensor pairs
    # Here we can select only distances where both sensors are in ARM_TRUNK_SENSORS
    # vuwb_arm = vuwb[:, pair_indices_for_arm_trunk]  # implement if needed
    ekf_data_arm = {
        "acc_world": acc_world_arm,
        "vel": vel_arm,
        "pos": pos_arm,
        "quat": quats_arm,
        "vuwb": vuwb   # for now keep all distances; can filter later if needed
    }

    save_path_arm = os.path.join(save_dir_arm, f"{base_name}_ekf_arm_trunk.pt")
    torch.save(ekf_data_arm, save_path_arm)
    print(f"Saved arm+trunk EKF data to {save_path_arm}")

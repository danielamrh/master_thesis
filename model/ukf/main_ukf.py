import torch
import os
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from model.ukf.ukf import UKF
from model.ekf.ik_solver_ekf import solve_fabrik, calculate_elbow_aa_from_gt_shoulder

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
from aitviewer.renderables.spheres import Spheres
from scipy.spatial.transform import Rotation as R

C.window_type = "pyqt5"

def smooth_data(data, cutoff=1.0, fs=100.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data

    smoothed_data = np.zeros_like(data_np)
    if data_np.ndim == 1:
        smoothed_data = filtfilt(b, a, data_np)
    else:
        for i in range(data_np.shape[1]):
            smoothed_data[:, i] = filtfilt(b, a, data_np[:, i])
            
    return torch.from_numpy(smoothed_data).float().to(data.device if isinstance(data, torch.Tensor) else 'cpu')

if __name__ == "__main__":
    
    # --- Configuration ---
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR
    SEQUENCE_INDEX = 0
    DOWNSAMPLE_RATE = 4
    TARGET_ARM = 'left' 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    DEBUG_ELBOW_ONLY = True
    
    print("Loading dataset...")
    original_data = torch.load(DATASET_PATH, map_location=DEVICE)
    print("Data loaded.")

    ACC_KEY, ORI_KEY, UWB_KEY, POSE_KEY, TRAN_KEY = 'acc', 'ori', 'vuwb', 'pose', 'tran'
    
    if TARGET_ARM == 'left':
        PELVIS_IDX, WRIST_IDX = 5, 0
        SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX = 16, 18, 20
        SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16-1, 18-1 
    else: # 'right'
        PELVIS_IDX, WRIST_IDX = 5, 1
        SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX = 17, 19, 21
        SHLDR_POSE_IDX, ELBOW_POSE_IDX = 17-1, 19-1

    print("Calibrating geometric model...")
    smpl_layer_cpu = SMPLLayer(model_type="smpl", gender="male", device='cpu')
    gt_poses_all_cpu = original_data[POSE_KEY][SEQUENCE_INDEX].cpu()
    gt_trans_cpu = original_data[TRAN_KEY][SEQUENCE_INDEX].cpu()
    
    p_p_init = gt_trans_cpu[0].clone()
    num_frames = gt_poses_all_cpu.shape[0]

    betas_cpu = torch.zeros(1, 10)
    _, gt_joints_t0 = smpl_layer_cpu(
        poses_body=gt_poses_all_cpu[0:1, 1:, :].flatten(start_dim=1),
        poses_root=gt_poses_all_cpu[0:1, 0, :],
        trans=gt_trans_cpu[0:1],
        betas=betas_cpu
    )
    gt_joints_t0 = gt_joints_t0.squeeze(0)

    p_shldr_t0 = gt_joints_t0[SHLDR_JOINT_IDX]
    p_elbow_t0 = gt_joints_t0[ELBOW_JOINT_IDX]
    p_wrist_t0 = gt_joints_t0[WRIST_JOINT_IDX]
    l1 = torch.linalg.norm(p_elbow_t0 - p_shldr_t0).item()
    l2 = torch.linalg.norm(p_wrist_t0 - p_elbow_t0).item()

    p_pelvis_t0 = gt_joints_t0[0]
    R_pelvis_t0_inv = R.from_rotvec(gt_poses_all_cpu[0, 0, :].numpy()).as_matrix().T
    s_offset_world = p_shldr_t0 - p_pelvis_t0
    s_offset = torch.from_numpy(R_pelvis_t0_inv @ s_offset_world.numpy()).float().to(DEVICE)
    print(f"Calibration complete: l1={l1:.3f}m, l2={l2:.3f}m")
    
    print("Pre-calculating ground truth joint path (memory-efficiently)...")
    gt_wrist_path_full = torch.zeros(num_frames, 3, device=DEVICE)
    for t in range(num_frames):
        _, gt_joints_frame = smpl_layer_cpu(
            poses_body=gt_poses_all_cpu[t:t+1, 1:, :].flatten(start_dim=1),
            poses_root=gt_poses_all_cpu[t:t+1, 0, :],
            trans=gt_trans_cpu[t:t+1],
            betas=betas_cpu
        )
        gt_wrist_path_full[t] = gt_joints_frame.squeeze(0)[WRIST_JOINT_IDX].to(DEVICE)

    del smpl_layer_cpu, gt_poses_all_cpu, gt_trans_cpu
    
    print("\nRunning Adaptive UKF-IK Pipeline...")
    aukf = UKF(device=DEVICE)
    acc_local_seq = original_data[ACC_KEY][SEQUENCE_INDEX]
    ori_matrix_seq = original_data[ORI_KEY][SEQUENCE_INDEX]
    uwb_dist_matrix = original_data[UWB_KEY][SEQUENCE_INDEX]
    
    print("Pre-filtering accelerometer data...")
    acc_local_seq_filtered = smooth_data(acc_local_seq.reshape(-1, 6 * 3), cutoff=8.0).reshape(num_frames, 6, 3)

    v_p_init = torch.zeros(3, device=DEVICE)
    p_w_init = p_p_init + s_offset + torch.tensor([0, -l1 -l2, 0], device=DEVICE, dtype=torch.float32)
    v_w_init = torch.zeros(3, device=DEVICE)
    aukf.set_initial(p_p_init=p_p_init.to(DEVICE), v_p_init=v_p_init, p_w_init=p_w_init, v_w_init=v_w_init)
    
    all_est_states = []
    ukf_wrist_errors = [] 

    for t in range(num_frames):
        acc_p_local, acc_w_local = acc_local_seq_filtered[t, PELVIS_IDX], acc_local_seq_filtered[t, WRIST_IDX]
        R_p, R_w = ori_matrix_seq[t, PELVIS_IDX], ori_matrix_seq[t, WRIST_IDX]
        
        # --- DEFINITIVE FIX: Removed incorrect gravity subtraction ---
        acc_p_world = (R_p @ acc_p_local.unsqueeze(1)).squeeze()
        acc_w_world = (R_w @ acc_w_local.unsqueeze(1)).squeeze()
        
        aukf.predict(acc_p_world, acc_w_world)
        
        dist_meas = uwb_dist_matrix[t, PELVIS_IDX, WRIST_IDX].item()
        if 0.2 < dist_meas < 2.0:
            accepted, nis = aukf.update_uwb_scalar_distance(dist_meas)
            if not accepted:
                # This print is useful for debugging NLOS
                # print(f"UWB measurement rejected at frame {t}. NIS: {nis:.2f}")
                pass

        aukf.update_zero_velocity_pelvis(acc_p_local)
        
        state, _ = aukf.get_state()
        all_est_states.append(state.clone())

        current_est_wrist_pos = state[9:12]
        current_gt_wrist_pos = gt_wrist_path_full[t]
        error = torch.linalg.norm(current_est_wrist_pos - current_gt_wrist_pos).item()
        ukf_wrist_errors.append(error)
    
    print("AUKF pipeline complete.")
    est_states = torch.stack(all_est_states)
    
    print("\nApplying post-processing and solving IK...")
    est_trans = est_states[:, 0:3].squeeze()
    est_wrist_pos_raw = est_states[:, 9:12].squeeze()
    est_wrist_pos_smoothed = smooth_data(est_wrist_pos_raw, cutoff=4.0)

    gt_poses_all = original_data[POSE_KEY][SEQUENCE_INDEX]
    gt_trans = original_data[TRAN_KEY][SEQUENCE_INDEX]
    
    offset = gt_trans[0] - est_trans[0]
    est_trans_aligned = est_trans + offset
    
    est_poses_body = torch.zeros(num_frames, 23 * 3, device=DEVICE)
    
    for t in range(num_frames):
        p_wrist_target = est_wrist_pos_smoothed[t]
        
        temp_pose = torch.zeros(23, 3, device=DEVICE)

        if DEBUG_ELBOW_ONLY:
            gt_poses_root_offset = 1
            gt_shoulder_aa = gt_poses_all[t, gt_poses_root_offset + SHLDR_POSE_IDX, :]
            R_shoulder_world = torch.from_numpy(R.from_rotvec(gt_shoulder_aa.cpu().numpy()).as_matrix()).float().to(DEVICE)
            
            R_pelvis_t_numpy = R.from_rotvec(gt_poses_all[t,0,:].cpu().numpy()).as_matrix()
            R_pelvis_t = torch.from_numpy(R_pelvis_t_numpy).float().to(DEVICE)
            p_shoulder_world = gt_trans[t] + (R_pelvis_t @ s_offset)

            p_elbow_initial = p_shoulder_world + R_shoulder_world @ torch.tensor([0.0, 0.0, -l1], device=DEVICE)
            initial_chain = torch.stack([p_shoulder_world, p_elbow_initial, p_wrist_target])
            solved_chain = solve_fabrik(initial_chain, p_wrist_target, torch.tensor([l1, l2], device=DEVICE))
            
            elbow_aa = calculate_elbow_aa_from_gt_shoulder(solved_chain[0], solved_chain[1], solved_chain[2], R_shoulder_world, side=TARGET_ARM)

            temp_pose[SHLDR_POSE_IDX] = gt_shoulder_aa
            temp_pose[ELBOW_POSE_IDX] = elbow_aa
        else:
            pass

        est_poses_body[t] = temp_pose.flatten()

    print("\nSetting up visualization...")
    v = Viewer()
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=DEVICE)
    
    gt_poses_root = gt_poses_all[:, 0, :]
    gt_poses_body = gt_poses_all[:, 1:, :].flatten(start_dim=1)

    seq_gt = SMPLSequence(smpl_layer=smpl_layer, 
                          poses_body=gt_poses_body[::DOWNSAMPLE_RATE], 
                          poses_root=gt_poses_root[::DOWNSAMPLE_RATE], 
                          trans=gt_trans[::DOWNSAMPLE_RATE], 
                          name="Ground Truth (Red)")
    seq_gt.color = (1.0, 0.2, 0.2, 1.0)
    v.scene.add(seq_gt)

    est_poses_full = gt_poses_body.clone().reshape(num_frames, 23, 3)
    est_poses_body_reshaped = est_poses_body.reshape(num_frames, 23, 3)
    est_poses_full[:, SHLDR_POSE_IDX] = est_poses_body_reshaped[:, SHLDR_POSE_IDX]
    est_poses_full[:, ELBOW_POSE_IDX] = est_poses_body_reshaped[:, ELBOW_POSE_IDX]
    
    seq_est = SMPLSequence(smpl_layer=smpl_layer, 
                           poses_body=est_poses_full.flatten(start_dim=1)[::DOWNSAMPLE_RATE],
                           poses_root=gt_poses_root[::DOWNSAMPLE_RATE], 
                           trans=est_trans_aligned[::DOWNSAMPLE_RATE],
                           name=f"My AUKF Estimate ({TARGET_ARM.title()} Arm)")
    seq_est.color = (0.2, 0.6, 1.0, 1.0)
    v.scene.add(seq_est)
    
    est_wrist_path = est_wrist_pos_smoothed.cpu().numpy()

    gt_wrist_spheres = Spheres(positions=gt_wrist_path_full[::DOWNSAMPLE_RATE].cpu().numpy(), radius=0.015, color=(1.0, 0.0, 1.0, 0.8))
    est_wrist_spheres = Spheres(positions=est_wrist_path[::DOWNSAMPLE_RATE], radius=0.015, color=(0.0, 1.0, 0.0, 0.8))
    v.scene.add(gt_wrist_spheres, est_wrist_spheres)

    print("Visualization ready. Press 'P' to play/pause.")
    v.run()

    print("\nPlotting UKF wrist position error...")
    plt.figure(figsize=(12, 6))
    plt.plot(ukf_wrist_errors)
    plt.title("UKF Wrist Position Error Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Error (meters)")
    plt.grid(True)
    plt.show()
    print("Done.")

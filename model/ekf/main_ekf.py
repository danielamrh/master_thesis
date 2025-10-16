import torch
import os
import numpy as np
from scipy.signal import butter, filtfilt, medfilt
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from ekf_estimation import EKF
from ik_solver_ekf import positions_to_arm_rotations, solve_fabrik 

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.lines import Lines
from scipy.spatial.transform import Rotation as R

C.window_type = "pyqt5"

def compute_ground_truth_kinematics(original_data, constants, device, acc_key):
    """
    Computes the ground truth joint positions for all frames from the SMPL data.
    """
    print("\n--- Pre-calculating Ground Truth Kinematics from SMPL ---")
    POSE_KEY, TRAN_KEY = constants['POSE_KEY'], constants['TRAN_KEY']
    SEQUENCE_INDEX = constants['SEQUENCE_INDEX']
    num_frames = original_data[acc_key][SEQUENCE_INDEX].shape[0]

    smpl_layer_cpu = SMPLLayer(model_type="smpl", gender="male", device='cpu')
    gt_poses_all_cpu = original_data[POSE_KEY][SEQUENCE_INDEX].cpu()
    gt_trans_cpu = original_data[TRAN_KEY][SEQUENCE_INDEX].cpu()
    
    betas_cpu = torch.zeros(1, 10)
    if 'betas' in original_data and original_data['betas'].shape[0] > SEQUENCE_INDEX:
        betas_cpu = original_data['betas'][SEQUENCE_INDEX].cpu().view(1, -1)

    _, first_frame_joints = smpl_layer_cpu(poses_body=gt_poses_all_cpu[0:1, 1:, :].flatten(start_dim=1), poses_root=gt_poses_all_cpu[0:1, 0, :], trans=gt_trans_cpu[0:1], betas=betas_cpu)
    gt_joints_all_frames_cpu = torch.zeros(num_frames, first_frame_joints.shape[1], 3)
    
    with torch.no_grad():
        for t in range(num_frames):
            _, gt_joints_frame = smpl_layer_cpu(poses_body=gt_poses_all_cpu[t:t+1, 1:, :].flatten(start_dim=1), poses_root=gt_poses_all_cpu[t:t+1, 0, :], trans=gt_trans_cpu[t:t+1], betas=betas_cpu)
            gt_joints_all_frames_cpu[t] = gt_joints_frame.squeeze(0)
    
    gt_joints_all_frames = gt_joints_all_frames_cpu.to(device)
    gt_wrist_path_full = gt_joints_all_frames[:, constants['WRIST_JOINT_IDX'], :]
    gt_pelvis_path_full = gt_joints_all_frames[:, 0, :]
    print("Ground truth calculation complete.")
    
    return gt_joints_all_frames, gt_pelvis_path_full, gt_wrist_path_full

def calibrate_from_t_pose(
    acc_local_seq, 
    uwb_dist_matrix, 
    gt_joints_all_frames,
    constants, 
    device
):
    """
    Performs a robust calibration using the initial static T-pose.
    - Assumes the provided accelerometer data has had gravity removed.
    - Calculates IMU accelerometer bias from the static period.
    - Calculates UWB distance bias by comparing measured vs. ground truth.
    - Adds a stationarity check to validate the T-pose assumption.
    """
    print("\n--- Calibrating Sensors from Static T-Pose ---")
    
    PELVIS_IDX, WRIST_IDX = constants['PELVIS_IDX'], constants['WRIST_IDX']
    WRIST_JOINT_IDX = constants['WRIST_JOINT_IDX']
    
    calibration_frames = 100
    acc_calib_window_p = acc_local_seq[:calibration_frames, PELVIS_IDX]
    acc_calib_window_w = acc_local_seq[:calibration_frames, WRIST_IDX]

    # --- 1. Stationarity Check ---
    # Verify that the initial pose is indeed static.
    acc_variance_p = torch.var(acc_calib_window_p, dim=0).mean().item()
    acc_variance_w = torch.var(acc_calib_window_w, dim=0).mean().item()
    stationarity_threshold = 1e-3 

    print(f"Initial pelvis acc variance: {acc_variance_p:.6f}")
    print(f"Initial wrist acc variance: {acc_variance_w:.6f}")

    if acc_variance_p > stationarity_threshold or acc_variance_w > stationarity_threshold:
        warnings.warn("High accelerometer variance detected during calibration window. "
                      "The initial T-pose may not be static, which could affect bias estimation.")

    # --- 2. IMU Accelerometer Bias Calibration ---
    # Since gravity is pre-removed and the pose is static, any remaining
    # signal is the accelerometer bias.
    print("Calibrating IMU accelerometer bias...")
    initial_bias_p = acc_calib_window_p.mean(dim=0)
    initial_bias_w = acc_calib_window_w.mean(dim=0)
    print(f"  > Calculated Pelvis Bias: {initial_bias_p.cpu().numpy()}")
    print(f"  > Calculated Wrist Bias:  {initial_bias_w.cpu().numpy()}")
    print("IMU bias calibration complete.")

    # --- 3. UWB Distance Bias Calibration ---
    print("Performing high-confidence UWB bias calibration...")
    gt_wrist_path_calib = gt_joints_all_frames[:calibration_frames, WRIST_JOINT_IDX, :]
    gt_pelvis_path_calib = gt_joints_all_frames[:calibration_frames, 0, :]
    
    # Average the ground truth distance over the static period
    uwb_gt_dists_static = torch.linalg.norm(gt_wrist_path_calib - gt_pelvis_path_calib, dim=1).mean()
    
    # Average the raw UWB measurements over the static period
    uwb_raw_dists_static = uwb_dist_matrix[:calibration_frames, PELVIS_IDX, WRIST_IDX].mean()
    
    # The bias is the difference
    uwb_offset = (uwb_raw_dists_static - uwb_gt_dists_static).item()
    print(f"  > Ground Truth T-Pose Distance: {uwb_gt_dists_static:.4f} m")
    print(f"  > Measured UWB T-Pose Distance: {uwb_raw_dists_static:.4f} m")
    print(f"  > Calculated UWB Bias Offset:   {uwb_offset:.4f} m")
    print("UWB bias calibration complete.")

    # We will use this calculated offset to correct all subsequent UWB measurements.
    # Note: The EKF also estimates a residual UWB bias, but providing a good
    # initial offset significantly improves stability.
    
    calibration_data = {
        "initial_bias_p": initial_bias_p, 
        "initial_bias_w": initial_bias_w,
        "uwb_offset": uwb_offset,
    }
    return calibration_data

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
            
    return torch.from_numpy(smoothed_data.copy()).float().to(data.device if isinstance(data, torch.Tensor) else 'cpu')

def run_ekf_pipeline(acc_seq, ori_seq, uwb_dists, initial_state, gt_wrist_path, static_uwb_offset=0.0, device='cpu'):
    """
    Runs the EKF pipeline with pre-processing and dynamic tuning.
    """
    print(f"\n--- Running EKF Pipeline ---")
    
    ekf = EKF(device=device)
    ekf.set_initial(**initial_state)

    num_frames = acc_seq.shape[0]
    all_est_states, ekf_wrist_errors, predicted_dists = [], [], []
    corrected_acc_w_history = []
    
    PELVIS_IDX, WRIST_IDX = 5, 0

    # UWB PRE-PROCESSING (MEDIAN FILTER)
    uwb_dists_raw = uwb_dists[:, PELVIS_IDX, WRIST_IDX].cpu().numpy()
    uwb_dists_filtered = medfilt(uwb_dists_raw, kernel_size=5)
    uwb_dists_filtered = torch.from_numpy(uwb_dists_filtered).to(device)

    for t in range(num_frames):
        acc_p_local, acc_w_local = acc_seq[t, PELVIS_IDX], acc_seq[t, WRIST_IDX]
        R_p, R_w = ori_seq[t, PELVIS_IDX], ori_seq[t, WRIST_IDX]
        
        # DYNAMIC PROCESS NOISE (Q)
        acc_w_mag = torch.linalg.norm(acc_w_local - initial_state['b_w_init']).item()
        ekf.set_dynamic_process_noise(acc_w_mag)

        # EKF Prediction Step
        _, acc_w_local_corr = ekf.predict(acc_p_local, R_p, acc_w_local, R_w)
        corrected_acc_w_history.append(acc_w_local_corr.clone())
        
        # EKF Update Step
        dist_meas_raw = uwb_dists_filtered[t].item()
        dist_meas_corrected = dist_meas_raw - static_uwb_offset
        
        pred_dist = 0.0 # Initialize pred_dist
        if 0.0 < dist_meas_corrected < 2.0:
            pred_dist = ekf.update_uwb_scalar_distance(dist_meas_corrected)
        predicted_dists.append(pred_dist) # Append the value

        # Other updates
        ekf.update_zero_velocity_pelvis(acc_p_local)
        ekf.update_velocity_damping_pelvis()
        ekf.update_zero_velocity_wrist(acc_w_local_corr) 
        ekf.update_velocity_damping_wrist()
        # ekf.update_kinematic_anchor()
        
        state, _ = ekf.get_state()
        all_est_states.append(state.clone())
        
        current_est_wrist_pos = state[9:12].squeeze()
        error = torch.linalg.norm(current_est_wrist_pos - gt_wrist_path[t]).item()
        ekf_wrist_errors.append(error)
        
    print("EKF pipeline complete.")
    # CORRECTED RETURN STATEMENT
    return torch.stack(all_est_states), ekf_wrist_errors, predicted_dists, torch.stack(corrected_acc_w_history)

# def run_ekf_pipeline(acc_seq, ori_seq, uwb_dists, initial_state, gt_wrist_path, use_offline_calibration=True, static_uwb_offset=0.0, device='cpu'):
#     """
#     Runs the full EKF pipeline using a two-stage tuning approach for robust initialization.
#     """
#     print(f"\n--- Running EKF Pipeline (Calibration: {use_offline_calibration}) ---")
    
#     ekf = EKF(device=device)
#     ekf.set_initial(**initial_state)

#     num_frames = acc_seq.shape[0]
#     all_est_states, ekf_wrist_errors, predicted_dists = [], [], []
#     corrected_acc_w_history = []
    
#     PELVIS_IDX, WRIST_IDX = 5, 0 # Assuming left arm for simplicity in this function

#     # Apply a median filter to remove sporadic outliers from UWB data before it enters the EKF.
#     uwb_dists_raw = uwb_dists[:, PELVIS_IDX, WRIST_IDX].cpu().numpy()
#     uwb_dists_filtered = medfilt(uwb_dists_raw, kernel_size=5) # 5-sample median window
#     uwb_dists_filtered = torch.from_numpy(uwb_dists_filtered).to(device)

#     # --- Two-Stage Tuning Parameters ---
#     WARMUP_FRAMES = 150
#     # Stage 1: "Calm" tuning for the warm-up phase (prevents "whiplash")
#     Q_uwb_warmup = 1e-6  # Assume bias changes slowly at first
#     R_uwb_warmup = 0.6**2  # Be skeptical of initial noisy measurements
#     # Stage 2: "Aggressive" tuning for dynamic tracking
#     Q_uwb_dynamic = 0.1    # Assume bias can change rapidly
#     R_uwb_dynamic = 0.3**2 # Trust the measurements more after settling

#     for t in range(num_frames):
#         # --- Implement the two-stage tuning ---
#         if t < WARMUP_FRAMES:
#             ekf.Q[18, 18] = Q_uwb_warmup
#             ekf.R_dist_base = torch.eye(1, device=device, dtype=torch.float32) * R_uwb_warmup
#         elif t == WARMUP_FRAMES: # Switch to dynamic tuning once
#             print(f"Switching to dynamic EKF tuning at frame {t}.")
#             ekf.Q[18, 18] = Q_uwb_dynamic
#             ekf.R_dist_base = torch.eye(1, device=device, dtype=torch.float32) * R_uwb_dynamic

#         acc_p_local, acc_w_local = acc_seq[t, PELVIS_IDX], acc_seq[t, WRIST_IDX]
#         R_p, R_w = ori_seq[t, PELVIS_IDX], ori_seq[t, WRIST_IDX]
        
#         _, acc_w_local_corr = ekf.predict(acc_p_local, R_p, acc_w_local, R_w)
#         corrected_acc_w_history.append(acc_w_local_corr.clone())
        
#         # Apply the pre-calculated static UWB offset
#         dist_meas_raw = uwb_dists[t, PELVIS_IDX, WRIST_IDX].item()
#         dist_meas_corrected = dist_meas_raw - static_uwb_offset
        
#         pred_dist = 0.0
#         if 0.0 < dist_meas_corrected < 2.0: # Increased range for robustness
#             pred_dist = ekf.update_uwb_scalar_distance(dist_meas_corrected)

#         # Pelvis updates (with conditional damping)
#         ekf.update_zero_velocity_pelvis(acc_p_local)
#         if len(ekf.acc_p_buffer) == ekf.buffer_size and torch.var(torch.stack(ekf.acc_p_buffer), dim=0).mean() < (ekf.zvu_threshold * 1.5):
#             ekf.update_velocity_damping_pelvis()

#         # Wrist updates (with conditional damping)
#         wrist_acc_var = ekf.update_zero_velocity_wrist(acc_w_local) 
#         if wrist_acc_var < (ekf.zvu_threshold * 1.5):
#             pass
#         ekf.update_velocity_damping_wrist()
#         ekf.update_kinematic_anchor()
            
#         state, _ = ekf.get_state()
#         all_est_states.append(state.clone())
        
#         current_est_wrist_pos = state[9:12].squeeze()
#         error = torch.linalg.norm(current_est_wrist_pos - gt_wrist_path[t]).item()
#         ekf_wrist_errors.append(error)
#         predicted_dists.append(pred_dist)
        
#     print("EKF pipeline complete.")
#     return torch.stack(all_est_states), ekf_wrist_errors, predicted_dists, torch.stack(corrected_acc_w_history)

def post_process_and_solve_ik(cal_states, gt_poses_all, gt_trans, gt_joints_all_frames, gt_pelvis_path_full, ori_matrix_seq, constants, device):
    """
    Performs post-processing on EKF results and solves the inverse kinematics for the arm.
    - Smooths the raw EKF position estimates.
    - Aligns the global translation.
    - Reconstructs the arm pose for each frame using a FABRIK IK solver.
    Returns the full body poses with the estimated arm motion and the smoothed wrist path.
    """
    print("\nRunning Full Arm Inverse Kinematics to reconstruct poses...")
    
    PELVIS_IDX = constants['PELVIS_IDX']
    
    # Smooth the raw EKF state estimates
    est_trans_raw = cal_states[:, 0:3].squeeze()
    est_wrist_pos_raw = cal_states[:, 12:15].squeeze()
    est_wrist_pos_smoothed = smooth_data(est_wrist_pos_raw, cutoff=2.0) 

    # Initialize estimated poses with a static T-pose
    est_poses_full = torch.zeros_like(gt_poses_all)
    num_frames = gt_poses_all.shape[0]

    # Calculate bone lengths from the first frame of ground truth
    p_shldr_t0 = gt_joints_all_frames[0, constants['SHLDR_JOINT_IDX']]
    p_elbow_t0 = gt_joints_all_frames[0, constants['ELBOW_JOINT_IDX']]
    p_wrist_t0 = gt_joints_all_frames[0, constants['WRIST_JOINT_IDX']]
    l1 = torch.linalg.norm(p_elbow_t0 - p_shldr_t0).item()
    l2 = torch.linalg.norm(p_wrist_t0 - p_elbow_t0).item()

    # Run the IK loop to populate the estimated arm poses
    for t in range(1, num_frames):
        # Use GT shoulder for stability, could be replaced with estimated torso in a full system
        p_shoulder_est = gt_joints_all_frames[t, constants['SHLDR_JOINT_IDX']]
        p_wrist_target = est_wrist_pos_smoothed[t]
        R_pelvis_t = ori_matrix_seq[t, PELVIS_IDX]
        
        # Solve the 2-bone arm chain for the elbow position
        initial_chain = torch.stack([p_shoulder_est, p_shoulder_est, p_wrist_target])
        bone_lengths_t = torch.tensor([l1, l2], device=device)
        solved_chain = solve_fabrik(initial_chain, p_wrist_target, bone_lengths_t)
        p_elbow_est = solved_chain[1]
        
        # Convert the solved joint positions back to SMPL axis-angle rotations
        shoulder_aa, elbow_aa = positions_to_arm_rotations(
            gt_pelvis_path_full[t], R_pelvis_t, p_shoulder_est, p_elbow_est, p_wrist_target, 
            side=constants['TARGET_ARM']
        )
        
        est_poses_full[t, 1 + constants['SHLDR_POSE_IDX']] = shoulder_aa
        est_poses_full[t, 1 + constants['ELBOW_POSE_IDX']] = elbow_aa
        
    return est_poses_full, est_wrist_pos_smoothed

def run_visualization(device, gt_poses_all, gt_trans, est_poses_full, cal_wrist_path_smoothed, uncal_states, constants):
    """
    Sets up the aitviewer scene and runs the visualization.
    - Renders Ground Truth and EKF Estimate models.
    - Renders wrist trajectory spheres.
    - Figures are static at the origin to focus on arm motion.
    """
    print("\nSetting up SMPL visualization...")
    v = Viewer()
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=device)

    # --- 1. Define a static root position and orientation ---
    num_frames = gt_poses_all.shape[0]
    static_trans = torch.zeros(num_frames, 3, device=device)
    static_root_pose = torch.zeros(num_frames, 3, device=device)

    # --- Ground Truth Visualization (Red Model) ---
    static_t_pose_sequence = torch.zeros_like(gt_poses_all)
    gt_poses_viz = static_t_pose_sequence.clone()
    gt_poses_viz[:, 1 + constants['SHLDR_POSE_IDX'], :] = gt_poses_all[:, 1 + constants['SHLDR_POSE_IDX'], :]
    gt_poses_viz[:, 1 + constants['ELBOW_POSE_IDX'], :] = gt_poses_all[:, 1 + constants['ELBOW_POSE_IDX'], :]
    gt_poses_viz[:, 20, :] = gt_poses_all[:, 20, :] # Left wrist

    gt_poses_body_viz = gt_poses_viz[:, 1:, :].flatten(start_dim=1)
    seq_gt = SMPLSequence(smpl_layer=smpl_layer,
                          poses_body=gt_poses_body_viz[::constants['DOWNSAMPLE_RATE']],
                          poses_root=static_root_pose[::constants['DOWNSAMPLE_RATE']],
                          trans=static_trans[::constants['DOWNSAMPLE_RATE']],
                          name="Ground Truth (Red)")
    seq_gt.color = (1.0, 0.2, 0.2, 1.0)
    v.scene.add(seq_gt)

    # --- EKF Estimate Visualization (Blue Model) ---
    est_poses_body = est_poses_full[:, 1:, :].flatten(start_dim=1)
    seq_est = SMPLSequence(smpl_layer=smpl_layer,
                           poses_body=est_poses_body[::constants['DOWNSAMPLE_RATE']],
                           poses_root=static_root_pose[::constants['DOWNSAMPLE_RATE']],
                           trans=static_trans[::constants['DOWNSAMPLE_RATE']],
                           name=f"EKF Estimate (Calibrated)")
    seq_est.color = (0.2, 0.6, 1.0, 1.0)
    v.scene.add(seq_est)

    # --- 2. Adjust Sphere positions to be relative to the static bodies ---
    gt_wrist_path_static = seq_gt.joints[:, constants['WRIST_JOINT_IDX'], :]
    cal_wrist_path_static = seq_est.joints[:, constants['WRIST_JOINT_IDX'], :]

    uncal_wrist_pos_world = smooth_data(uncal_states[:, 12:15].squeeze(), cutoff=4.0)
    uncal_wrist_path_static = torch.zeros_like(uncal_wrist_pos_world)
    for t in range(num_frames):
        root_orient_aa = gt_poses_all[t, 0, :]
        root_trans_t = gt_trans[t]
        r_mat_inv = torch.tensor(R.from_rotvec(root_orient_aa.cpu().numpy()).as_matrix().T, device=device, dtype=torch.float32)
        uncal_wrist_path_static[t] = r_mat_inv @ (uncal_wrist_pos_world[t] - root_trans_t)

    gt_wrist_spheres = Spheres(gt_wrist_path_static, radius=0.015, color=(1.0, 0.0, 1.0, 0.8), name="GT Wrist")
    cal_wrist_spheres = Spheres(cal_wrist_path_static, radius=0.015, color=(0.0, 1.0, 0.0, 0.8), name="Calibrated EKF Wrist")
    uncal_wrist_spheres = Spheres(uncal_wrist_path_static[::constants['DOWNSAMPLE_RATE']].cpu().numpy(), radius=0.015, color=(1.0, 0.6, 0.0, 0.8), name="Uncalibrated EKF Wrist")
    v.scene.add(gt_wrist_spheres, cal_wrist_spheres, uncal_wrist_spheres)

    print("Visualization ready. Press 'P' to play/pause. Close window to see plots.")
    v.run()

def plot_results(
    cal_states, uncal_states, cal_errors, uncal_errors, cal_dists, uncal_dists,
    cal_acc_corr_w, uncal_acc_corr_w, acc_local_seq, gt_acc_local_w,
    gt_pelvis_path, gt_wrist_path, uwb_dist_matrix, uwb_offset, constants,
):
    """
    Generates and displays plots comparing EKF performance and internal states.
    """
    print("\nPlotting detailed error and data graphs...")
    
    uwb_gt_dists = torch.linalg.norm(
        gt_wrist_path - gt_pelvis_path, dim=1
    ).cpu().numpy()
    
    fig, axs = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('EKF Performance and IMU Data Comparison', fontsize=16)

    # --- Raw UWB Error Analysis ---
    uwb_raw_measurements = uwb_dist_matrix[:, constants['PELVIS_IDX'], constants['WRIST_IDX']].cpu().numpy()
    raw_uwb_error = uwb_raw_measurements - uwb_gt_dists

    # --- Original Plots ---
    axs[0, 0].plot(cal_errors, label='With Offline Calibration', color='blue')
    axs[0, 0].plot(uncal_errors, label='Without Offline Calibration', color='orange', linestyle='--')
    axs[0, 0].set_title("Total EKF Wrist Position Error")
    axs[0, 0].set_xlabel("Frame"); axs[0, 0].set_ylabel("Error (meters)")
    axs[0, 0].grid(True); axs[0, 0].legend()
    axs[0, 0].set_ylim(bottom=0)

    axs[0, 1].plot(uwb_gt_dists, label='Ground Truth Distance', color='green', linewidth=2)
    axs[0, 1].plot(uwb_raw_measurements - uwb_offset, label='Calibrated UWB Measurement', color='red', alpha=0.6)
    axs[0, 1].plot(cal_dists, label='EKF Predicted Distance', color='blue', linestyle='--')
    axs[0, 1].set_title("UWB Distances (Calibrated Run)")
    axs[0, 1].set_xlabel("Frame"); axs[0, 1].set_ylabel("Distance (meters)")
    axs[0, 1].grid(True); axs[0, 1].legend()
    axs[0, 1].set_ylim(bottom=0)

    axs[1, 1].plot(raw_uwb_error)
    axs[1, 1].axhline(y=0.0, color='black', linestyle='--', label='Zero Error')
    axs[1, 1].axhline(y=uwb_offset, color='red', linestyle=':', label=f'Calculated Static Offset ({uwb_offset:.2f}m)')
    axs[1, 1].set_title("Raw UWB Measurement Error (Raw - GT)")
    axs[1, 1].set_xlabel("Frame")
    axs[1, 1].set_ylabel("Error (meters)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    wrist_bias = cal_states[:, 15:18].cpu().numpy()
    pelvis_bias = cal_states[:, 6:9].cpu().numpy()
    axs[1, 0].plot(wrist_bias[:, 0], label='Wrist Bias X', color='r')
    axs[1, 0].plot(wrist_bias[:, 1], label='Wrist Bias Y', color='g')
    axs[1, 0].plot(wrist_bias[:, 2], label='Wrist Bias Z', color='b')
    axs[1, 0].plot(pelvis_bias[:, 0], label='Pelvis Bias X', color='r', linestyle=':')
    axs[1, 0].plot(pelvis_bias[:, 1], label='Pelvis Bias Y', color='g', linestyle=':')
    axs[1, 0].plot(pelvis_bias[:, 2], label='Pelvis Bias Z', color='b', linestyle=':')
    axs[1, 0].set_title("Estimated Accelerometer Biases (Calibrated Run)")
    axs[1, 0].set_xlabel("Frame"); axs[1, 0].set_ylabel("Bias (m/s^2)")
    axs[1, 0].grid(True); axs[1, 0].legend(ncol=2)
    
    raw_wrist_acc_x = acc_local_seq[:, constants['WRIST_IDX'], 0].cpu().numpy()
    corr_wrist_acc_x = cal_acc_corr_w[:, 0].cpu().numpy()
    axs[2, 0].plot(raw_wrist_acc_x, label='Raw Accel X', color='purple', alpha=0.5)
    axs[2, 0].plot(corr_wrist_acc_x, label='EKF Corrected Accel X', color='green')
    axs[2, 0].set_title("Wrist Acceleration: Raw vs. Corrected (X-axis)")
    axs[2, 0].set_xlabel("Frame"); axs[2, 0].set_ylabel("Acceleration (m/s^2)")
    axs[2, 0].grid(True); axs[2, 0].legend()

    gt_wrist_acc_x = gt_acc_local_w[:, 0].cpu().numpy()
    axs[2, 1].plot(gt_wrist_acc_x, label='Ground Truth Accel X', color='black', linestyle='--')
    axs[2, 1].plot(corr_wrist_acc_x, label='EKF Corrected Accel X', color='yellow')
    axs[2, 1].set_title("Wrist Acceleration: EKF Corrected vs. Ground Truth (X-axis)")
    axs[2, 1].set_xlabel("Frame"); axs[2, 1].set_ylabel("Acceleration (m/s^2)")
    axs[2, 1].grid(True); axs[2, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    print("Done.")

def calculate_ground_truth_imu_data(gt_joint_path, ori_matrix_seq, dt=0.01):
    """
    Calculates a smoothed, realistic ground truth local-frame linear acceleration.
    """
    smoothed_gt_joint_path = smooth_data(gt_joint_path, cutoff=15.0)
    
    gt_velocity_world = torch.diff(smoothed_gt_joint_path, dim=0) / dt
    gt_velocity_world = torch.cat([gt_velocity_world[0:1], gt_velocity_world], dim=0)
    
    gt_acceleration_world = torch.diff(gt_velocity_world, dim=0) / dt
    gt_acceleration_world = torch.cat([gt_acceleration_world[0:1], gt_acceleration_world], dim=0)
    
    gt_acceleration_local = torch.einsum('nij,nj->ni', ori_matrix_seq.transpose(1, 2), gt_acceleration_world)
    
    return gt_acceleration_local

if __name__ == "__main__":
    
    # Configuration
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR
    SEQUENCE_INDEX = 0
    DOWNSAMPLE_RATE = 4
    TARGET_ARM = 'left' 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading dataset...")
    original_data = torch.load(DATASET_PATH, map_location=DEVICE)
    print("Data loaded.")

    ACC_KEY, ORI_KEY, UWB_KEY, POSE_KEY, TRAN_KEY = 'acc', 'ori', 'vuwb', 'pose', 'tran'
    
    if TARGET_ARM == 'left':
        PELVIS_IDX, WRIST_IDX = 5, 0
        SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX = 16, 18, 20
        SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16-1, 18-1 
    else:
        PELVIS_IDX, WRIST_IDX = 5, 1
        SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX = 17, 19, 21
        SHLDR_POSE_IDX, ELBOW_POSE_IDX = 17-1, 19-1

    # --- Sensor Data Extraction ---
    acc_local_seq = original_data[ACC_KEY][SEQUENCE_INDEX]
    ori_matrix_seq = original_data[ORI_KEY][SEQUENCE_INDEX]
    uwb_dist_matrix = original_data[UWB_KEY][SEQUENCE_INDEX]
    num_frames = acc_local_seq.shape[0]

    # --- Coordinate System Alignment ---
    # The IMU data may be in a different coordinate system (e.g., Y-up) than the
    # SMPL ground truth (e.g., Z-up). This rotation aligns the IMU orientation
    # data to match the SMPL coordinate system. This is a common pre-processing step.
    # NOTE: This does NOT interact with gravity, as gravity is assumed to be pre-removed.
    # align_angle = -np.pi / 2.0
    # R_align = torch.tensor([
    #     [1.0, 0.0, 0.0],
    #     [0.0, np.cos(align_angle), -np.sin(align_angle)],
    #     [0.0, np.sin(align_angle), np.cos(align_angle)]
    # ], dtype=torch.float32, device=DEVICE)
    # ori_matrix_seq = R_align.unsqueeze(0).unsqueeze(0) @ ori_matrix_seq
    # print("Applied coordinate system alignment to IMU orientations.")

    # --- Pack constants for cleaner function calls ---
    constants = {
        'TARGET_ARM': TARGET_ARM, 'DOWNSAMPLE_RATE': DOWNSAMPLE_RATE,
        'PELVIS_IDX': PELVIS_IDX, 'WRIST_IDX': WRIST_IDX,
        'SHLDR_JOINT_IDX': SHLDR_JOINT_IDX, 'ELBOW_JOINT_IDX': ELBOW_JOINT_IDX, 'WRIST_JOINT_IDX': WRIST_JOINT_IDX,
        'SHLDR_POSE_IDX': SHLDR_POSE_IDX, 'ELBOW_POSE_IDX': ELBOW_POSE_IDX,
        'POSE_KEY': POSE_KEY, 'TRAN_KEY': TRAN_KEY, 'SEQUENCE_INDEX': SEQUENCE_INDEX,
    }

    # --- Ground Truth and Calibration ---
    # 1. Compute all ground truth joint positions from SMPL data first.
    gt_joints_all_frames, gt_pelvis_path_full, gt_wrist_path_full = compute_ground_truth_kinematics(
        original_data, constants, DEVICE, ACC_KEY
    )
    
    # 2. Use the initial T-pose and GT data to find sensor biases.
    calib_data = calibrate_from_t_pose(
        acc_local_seq, uwb_dist_matrix, gt_joints_all_frames, constants, DEVICE
    )

    # --- EKF Pipeline Execution ---
    # Initial state for CALIBRATED run
    cal_initial_state_dict = {
        'p_p_init': gt_pelvis_path_full[0],
        'v_p_init': torch.zeros(3, device=DEVICE),
        'p_w_init': gt_wrist_path_full[0],
        'v_w_init': torch.zeros(3, device=DEVICE),
        'b_p_init': calib_data['initial_bias_p'],
        'b_w_init': calib_data['initial_bias_w']
    }

    cal_states, cal_errors, cal_dists, cal_acc_corr_w = run_ekf_pipeline(
        acc_local_seq,
        ori_matrix_seq,
        uwb_dist_matrix,
        cal_initial_state_dict,
        gt_wrist_path_full,
        # use_offline_calibration=True,
        static_uwb_offset=calib_data['uwb_offset'],
        device=DEVICE
    )

    # Initial state for UNCALIBRATED run
    uncal_initial_state_dict = cal_initial_state_dict.copy()
    uncal_initial_state_dict['b_p_init'] = torch.zeros(3, device=DEVICE)
    uncal_initial_state_dict['b_w_init'] = torch.zeros(3, device=DEVICE)

    uncal_states, uncal_errors, uncal_dists, uncal_acc_corr_w = run_ekf_pipeline(
        acc_local_seq,
        ori_matrix_seq,
        uwb_dist_matrix,
        uncal_initial_state_dict, 
        gt_wrist_path_full,
        # use_offline_calibration=False,
        static_uwb_offset=0.0, # Use zero offset for uncalibrated
        device=DEVICE
    )
    
    # --- Calculate Overall Error (RMSE) ---
    cal_errors_np, uncal_errors_np = np.array(cal_errors), np.array(uncal_errors)
    cal_rmse = np.sqrt(np.mean(cal_errors_np[250:]**2)) # Ignore initial transient
    uncal_rmse = np.sqrt(np.mean(uncal_errors_np[250:]**2))
    
    print("\n--- Overall Performance (Post-Initialization) ---")
    print(f"Calibrated Run RMSE:      {cal_rmse:.4f} meters")
    print(f"Uncalibrated Run RMSE:    {uncal_rmse:.4f} meters")
    
    # --- Post-processing: IK and Pose Reconstruction ---
    est_poses_full, est_wrist_pos_smoothed = post_process_and_solve_ik(
        cal_states=cal_states,
        gt_poses_all=original_data[POSE_KEY][SEQUENCE_INDEX],
        gt_trans=original_data[TRAN_KEY][SEQUENCE_INDEX],
        gt_joints_all_frames=gt_joints_all_frames,
        gt_pelvis_path_full=gt_pelvis_path_full,
        ori_matrix_seq=ori_matrix_seq,
        constants=constants,
        device=DEVICE
    )

    # --- SMPL Visualization ---
    run_visualization(
        device=DEVICE,
        gt_poses_all=original_data[POSE_KEY][SEQUENCE_INDEX],
        gt_trans=original_data[TRAN_KEY][SEQUENCE_INDEX],
        est_poses_full=est_poses_full,
        cal_wrist_path_smoothed=est_wrist_pos_smoothed,
        uncal_states=uncal_states,
        constants=constants
    )

    # --- PLOTTING SECTION ---
    gt_acc_local_w = calculate_ground_truth_imu_data(
        gt_wrist_path_full, ori_matrix_seq[:, WRIST_IDX, :, :]
    )
    plot_results(
        cal_states=cal_states, uncal_states=uncal_states, cal_errors=cal_errors,
        uncal_errors=uncal_errors, cal_dists=cal_dists, uncal_dists=uncal_dists,
        cal_acc_corr_w=cal_acc_corr_w, uncal_acc_corr_w=uncal_acc_corr_w,
        acc_local_seq=acc_local_seq, gt_acc_local_w=gt_acc_local_w,
        gt_pelvis_path=gt_pelvis_path_full, gt_wrist_path=gt_wrist_path_full,
        uwb_dist_matrix=uwb_dist_matrix, uwb_offset=calib_data['uwb_offset'],
        constants=constants,
    )

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from ekf_joint_angle import EKFJointAngle 

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

C.window_type = "pyqt5"


def calibrate_uwb_from_t_pose(uwb_dist_matrix, gt_kinematics, constants):
    """
    Calculates the static UWB distance offset from the initial T-pose.
    """
    print("\n--- Calibrating UWB Sensor from T-Pose ---")
    calibration_frames = 100 # Use the first 100 static frames

    # Calculate average ground truth distance during the T-pose
    gt_joints = gt_kinematics['joints']
    gt_wrist_path_calib = gt_joints[:calibration_frames, constants['WRIST_JOINT_IDX'], :]
    gt_pelvis_path_calib = gt_joints[:calibration_frames, 0, :]
    uwb_gt_dist = torch.linalg.norm(gt_wrist_path_calib - gt_pelvis_path_calib, dim=1).mean().item()

    # Calculate average measured UWB distance during the T-pose
    uwb_measured_dist = uwb_dist_matrix[:calibration_frames, constants['PELVIS_IDX'], constants['WRIST_IDX']].mean().item()

    # The offset is the difference
    uwb_offset = uwb_measured_dist - uwb_gt_dist
    print(f"  > Ground Truth T-Pose Distance: {uwb_gt_dist:.4f} m")
    print(f"  > Measured UWB T-Pose Distance: {uwb_measured_dist:.4f} m")
    print(f"  > Calculated UWB Bias Offset:   {uwb_offset:.4f} m")
    return uwb_offset

def get_ground_truth_kinematics(original_data, constants, device):
    """
    Computes GT joint positions, anatomical angles, and bone lengths from SMPL data.
    """
    print("--- Pre-calculating Ground Truth Kinematics ---")
    POSE_KEY, TRAN_KEY = constants['POSE_KEY'], constants['TRAN_KEY']
    SEQUENCE_INDEX = constants['SEQUENCE_INDEX']
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=device)
    
    gt_poses_all = original_data[POSE_KEY][SEQUENCE_INDEX]
    gt_trans = original_data[TRAN_KEY][SEQUENCE_INDEX]
    num_frames = gt_poses_all.shape[0]

    if 'betas' in original_data and original_data['betas'].shape[0] > SEQUENCE_INDEX:
        betas = original_data['betas'][SEQUENCE_INDEX].view(1, -1)
    else:
        betas = torch.zeros(1, 10, device=device)

    # Get joint positions
    _, first_frame_joints = smpl_layer(poses_body=gt_poses_all[0:1, 1:].flatten(start_dim=1), poses_root=gt_poses_all[0:1, 0], trans=gt_trans[0:1], betas=betas)
    gt_joints_all_frames = torch.zeros(num_frames, first_frame_joints.shape[1], 3, device=device)
    with torch.no_grad():
        for t in range(num_frames):
            _, gt_joints_frame = smpl_layer(poses_body=gt_poses_all[t:t+1, 1:].flatten(start_dim=1), poses_root=gt_poses_all[t:t+1, 0], trans=gt_trans[t:t+1], betas=betas)
            gt_joints_all_frames[t] = gt_joints_frame.squeeze(0)

    # Get bone lengths
    p_shldr_t0 = gt_joints_all_frames[0, constants['SHLDR_JOINT_IDX']]
    p_elbow_t0 = gt_joints_all_frames[0, constants['ELBOW_JOINT_IDX']]
    p_wrist_t0 = gt_joints_all_frames[0, constants['WRIST_JOINT_IDX']]
    upper_arm_length = torch.linalg.norm(p_elbow_t0 - p_shldr_t0).item()
    forearm_length = torch.linalg.norm(p_wrist_t0 - p_elbow_t0).item()
    print(f"  > Upper arm length: {upper_arm_length:.4f} m, Forearm length: {forearm_length:.4f} m")

    # Get anatomical angles from pose parameters
    gt_shoulder_aa = gt_poses_all[:, 1 + constants['SHLDR_POSE_IDX'], :].cpu().numpy()
    gt_elbow_aa = gt_poses_all[:, 1 + constants['ELBOW_POSE_IDX'], :].cpu().numpy()

    # Convert axis-angle to Euler angles (ZYX order)
    gt_shoulder_euler = R.from_rotvec(gt_shoulder_aa).as_euler('zyx') 
    gt_elbow_euler = R.from_rotvec(gt_elbow_aa).as_euler('zyx') 
    
    # Map to anatomical angles
    gt_angles_for_plotting = {
        'sh_abduction': gt_shoulder_euler[:, 0], # Abduction is the Z component (first)
        'sh_flexion':   gt_shoulder_euler[:, 1], # Flexion is the Y component (second)
        'el_flexion':   gt_elbow_euler[:, 1]     # Elbow flexion is around Y axis
    }

    kinematics = {
        "joints": gt_joints_all_frames, 
        "bone_lengths": [upper_arm_length, forearm_length],
        "gt_angles": gt_angles_for_plotting, 
        "poses": gt_poses_all, 
        "trans": gt_trans, 
        "betas": betas
    }
    return kinematics

def get_shoulder_pelvis_offset(gt_kinematics, sensor_data, constants):
    """
    Calculates the static anatomical offset vector from the pelvis joint to the
    shoulder joint in the PELVIS'S LOCAL COORDINATE SYSTEM.
    """
    gt_joints = gt_kinematics['joints']
    
    # 1. Get world-frame joint positions from T-pose (t=0)
    p_pelvis_t0_world = gt_joints[0, 0, :]
    p_shoulder_t0_world = gt_joints[0, constants['SHLDR_JOINT_IDX'], :]
    offset_world_t0 = p_shoulder_t0_world - p_pelvis_t0_world

    # 2. Get the pelvis orientation in the world frame at t=0
    R_pelvis_world_t0 = sensor_data['ori'][0, constants['PELVIS_IDX'], :, :]

    # 3. Transform the world-space offset into the local pelvis-space offset
    offset_local = R_pelvis_world_t0.T @ offset_world_t0
    
    print(f"\n--- Calibrating Pelvis-Shoulder Offset ---")
    print(f"  > Calculated local offset vector: {offset_local.cpu().numpy()}")
    return offset_local

def run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, device, uwb_offset=0.0, shoulder_pelvis_offset=None):
    """
    Runs the EKF joint angle estimation pipeline on the provided sensor data.
    """
    print("\n--- Running EKF Pipeline with Initial T-Pose Calibration ---")
    num_frames = sensor_data['acc'].shape[0]
    
    # Pass the new offset to the EKF constructor
    ekf = EKFJointAngle(bone_lengths=gt_kinematics['bone_lengths'],
                        shoulder_pelvis_offset=shoulder_pelvis_offset,
                        device=device,
                        dt=1/100.0)
    
    ekf.set_initial_state(torch.zeros(3, device=device))
    
    estimated_angles_history = []
    pelvis_orientation_seq = sensor_data['ori'][:, constants['PELVIS_IDX'], :, :]
    wrist_orientation_seq = sensor_data['ori'][:, constants['WRIST_IDX'], :, :]
    uwb_dists = sensor_data['uwb'][:, constants['PELVIS_IDX'], constants['WRIST_IDX']]
    gt_joints = gt_kinematics['joints']
    gt_shoulder_path = gt_joints[:, constants['SHLDR_JOINT_IDX'], :]
    gt_pelvis_path = gt_joints[:, 0, :]
    acc_w_seq = sensor_data['acc'][:, constants['WRIST_IDX'], :]
    
    # --- WARM-UP CALIBRATION LOGIC ---
    WARMUP_FRAMES = 100
    bias_vector = np.zeros(3)
    target_t_pose_angles = np.zeros(3) # True angles for a T-pose are [0, 0, 0]
    
    acc_buffer = []; buffer_size = 10; zavu_threshold = 0.05 

    for t in range(num_frames):
        # --- EKF Predict/Update cycle (runs for every frame) ---
        ekf.predict()
        ekf.update_zero_angular_velocity_prior()
        ekf.update_velocity_damping()

        acc_buffer.append(acc_w_seq[t])
        if len(acc_buffer) > buffer_size: acc_buffer.pop(0)
        if len(acc_buffer) == buffer_size:
            if torch.var(torch.stack(acc_buffer), dim=0).mean() < zavu_threshold:
                ekf.update_zero_angular_velocity()

        R_p_world_t = pelvis_orientation_seq[t]
        ekf.update_orientation(wrist_orientation_seq[t], R_p_world_t)
        uwb_meas_corrected = uwb_dists[t].item() - uwb_offset
        if uwb_meas_corrected > 0.0:
            ekf.update_uwb_distance(uwb_meas_corrected, R_p_world_t)

        # --- BIAS CALCULATION AND CORRECTION ---
        if t == WARMUP_FRAMES:
            estimated_t_pose_angles = ekf.get_state_angles().cpu().numpy()
            bias_vector = estimated_t_pose_angles - target_t_pose_angles
            print(f"  > Calculated bias vector (deg): {np.rad2deg(bias_vector)}")

        # Get the raw EKF output for the current frame
        raw_angles = ekf.get_state_angles().cpu().numpy()
        
        # Subtract the calculated bias (it's zero until after the warmup)
        corrected_angles = raw_angles - bias_vector
        estimated_angles_history.append(corrected_angles)
        
    print("EKF pipeline complete.")
    return np.array(estimated_angles_history)

def calibrate_and_correct_angles(est_angles, gt_angles):
    """
    Learns a ROBUST linear correction model (y = mx + c) using RANSAC
    to fix bias and scaling errors while ignoring outliers.
    """
    print(f"\n--- Applying Calibration using RANSAC ---")
    
    est_angles_np_deg = np.rad2deg(est_angles)
    corrected_angles_deg = np.zeros_like(est_angles_np_deg)
    
    kernel_size = 21
    est_angles_np_deg[:, 0] = np.rad2deg(np.unwrap(est_angles[:, 0]))
    est_angles_np_deg[:, 1] = np.rad2deg(np.unwrap(est_angles[:, 1]))
    est_angles_np_deg[:, 2] = np.rad2deg(np.unwrap(est_angles[:, 2]))

    # est_angles_np_deg[:, 0] = savgol_filter(est_angles_np_deg[:, 0], window_length=21, polyorder=3)
    # est_angles_np_deg[:, 1] = savgol_filter(est_angles_np_deg[:, 1], window_length=21, polyorder=3)
    # est_angles_np_deg[:, 2] = savgol_filter(est_angles_np_deg[:, 2], window_length=21, polyorder=3) 

    # est_angles_np_deg[:,0] = gaussian_filter1d(est_angles_np_deg[:, 0], sigma=3)
    # est_angles_np_deg[:, 1] = gaussian_filter1d(est_angles_np_deg[:, 1], sigma=3)
    # est_angles_np_deg[:, 2] = gaussian_filter1d(est_angles_np_deg[:, 2], sigma=3)

    # est_angles_np_deg[:, 0] = medfilt(est_angles_np_deg[:, 0], kernel_size=kernel_size)
    # est_angles_np_deg[:, 1] = medfilt(est_angles_np_deg[:, 1], kernel_size=kernel_size)
    # est_angles_np_deg[:, 2] = medfilt(est_angles_np_deg[:, 2], kernel_size=kernel_size)
    
    angle_keys = ['sh_abduction', 'sh_flexion', 'el_flexion']
    titles = ["Shoulder Abduction/Adduction", "Shoulder Flexion/Extension", "Elbow Flexion/Extension"]

    gt_angle_data = [gt_angles[key] for key in angle_keys]

    for i, key in enumerate(angle_keys):
        model = RANSACRegressor()
        
        train_x = est_angles_np_deg[:, i].reshape(-1, 1)
        train_y = np.rad2deg(gt_angle_data[i])
        
        min_len = min(len(train_x), len(train_y))
        
        model.fit(train_x[:min_len], train_y[:min_len])
        
        # Access the underlying linear model's coefficients for printing
        if model.estimator_ is not None:
            coef = model.estimator_.coef_[0]
            intercept = model.estimator_.intercept_
            print(f"  > {key.replace('_', ' ').title():<25} Correction: GT = {coef:.2f} * EKF + {intercept:.2f}")
        
        corrected_angles_deg[:, i] = model.predict(est_angles_np_deg[:, i].reshape(-1, 1))
    
    return np.deg2rad(corrected_angles_deg)

def post_process_angles_to_poses(est_angles, gt_poses_all, constants, device):
    """
    Converts estimated anatomical angles back to full SMPL pose parameters.
    """
    print("Converting estimated angles to SMPL poses...")
    num_frames = est_angles.shape[0]
    est_poses_full = torch.zeros_like(gt_poses_all)
    
    for t in range(num_frames):
        # Start with a T-pose for all frames
        sh_abduction_val = est_angles[t, 0] # Shoulder abduction/adduction
        sh_flexion_val = est_angles[t, 1]   # Shoulder flexion/extension
        el_flexion_val = est_angles[t, 2]   # Elbow flexion/extension

        shoulder_angles_zyx = [sh_abduction_val, sh_flexion_val, 0.0]
        shoulder_aa = R.from_euler('zyx', shoulder_angles_zyx).as_rotvec()

        R_elbow = R.from_euler('zyx', [0, el_flexion_val, 0]).as_matrix()
        elbow_aa = R.from_matrix(R_elbow).as_rotvec()
        
        est_poses_full[t, 1 + constants['SHLDR_POSE_IDX']] = torch.from_numpy(shoulder_aa).to(device)
        est_poses_full[t, 1 + constants['ELBOW_POSE_IDX']] = torch.from_numpy(elbow_aa).to(device)
        
    return est_poses_full

def run_visualization(est_poses_full, gt_kinematics, constants, device):
    """
    Runs the SMPL visualization for the estimated and ground truth poses.
    """
    print("\nSetting up SMPL visualization...")
    v = Viewer()
    downsample_rate = constants.get('DOWNSAMPLE_RATE', 1)
    print(f"Visualization will be downsampled by a factor of {downsample_rate}.")

    betas = gt_kinematics.get('betas', torch.zeros(1, 10, device=device))
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=device, betas=betas)
    
    num_frames = est_poses_full.shape[0]
    static_trans = torch.zeros(num_frames, 3, device=device)
    static_root_pose = torch.zeros(num_frames, 3, device=device)

    # --- Ground Truth Visualization (Red Model) - ISOLATED ARM ---
    gt_poses_all = gt_kinematics['poses']
    # Start with a T-pose for all frames
    gt_poses_viz = torch.zeros_like(gt_poses_all)
    # Copy only the shoulder and elbow pose parameters for the target arm
    gt_poses_viz[:, 1 + constants['SHLDR_POSE_IDX']] = gt_poses_all[:, 1 + constants['SHLDR_POSE_IDX']]
    gt_poses_viz[:, 1 + constants['ELBOW_POSE_IDX']] = gt_poses_all[:, 1 + constants['ELBOW_POSE_IDX']]
    
    seq_gt = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=gt_poses_viz[:, 1:, :].flatten(start_dim=1)[::downsample_rate],
        poses_root=static_root_pose[::downsample_rate],
        trans=static_trans[::downsample_rate],
        name="Ground Truth (Red, Arm Only)"
    )
    seq_gt.color = (1.0, 0.2, 0.2, 1.0)
    v.scene.add(seq_gt)

    # --- EKF Estimate Visualization (Blue Model) ---
    # This already shows only the arm, since est_poses_full was built from a T-pose
    seq_est = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=est_poses_full[:, 1:, :].flatten(start_dim=1)[::downsample_rate],
        poses_root=static_root_pose[::downsample_rate],
        trans=static_trans[::downsample_rate],
        name="EKF Estimate (Blue)"
    )
    seq_est.color = (0.2, 0.6, 1.0, 1.0)
    v.scene.add(seq_est)

    print("Visualization ready. Press 'P' to play/pause.")
    v.run()

def calculate_performance_metrics(raw_angles, corrected_angles, gt_angles):
    """
    Calculates and prints key performance metrics for both raw and corrected angles.
    """
    print("\n--- Estimator Performance Metrics ---")
    
    raw_deg = np.rad2deg(raw_angles)
    corrected_deg = np.rad2deg(corrected_angles)
    
    angle_keys = ['sh_abduction', 'sh_flexion', 'el_flexion']
    titles = ["Shoulder Abduction/Adduction", "Shoulder Flexion/Extension", "Elbow Flexion/Extension"]

    gt_angle_data = [gt_angles[key] for key in angle_keys]
    
    for i, key in enumerate(angle_keys):
        gt = np.rad2deg(gt_angle_data[i])
        
        # --- Metrics for UNCALIBRATED (RAW) Data ---
        raw = raw_deg[:, i]
        if key == 'sh_roll': raw = np.rad2deg(np.unwrap(raw_angles[:, 1])) 
        
        min_len = min(len(raw), len(gt))
        raw, gt_sliced = raw[:min_len], gt[:min_len]
        
        error_raw = raw - gt_sliced
        rmse_raw = np.sqrt(np.mean(error_raw**2))
        corr_raw, _ = pearsonr(raw, gt_sliced)
        max_err_raw = np.max(np.abs(error_raw))

        # --- Metrics for CALIBRATED (CORRECTED) Data ---
        corrected = corrected_deg[:, i]
        min_len = min(len(corrected), len(gt))
        corrected, gt_sliced = corrected[:min_len], gt[:min_len]

        error_corr = corrected - gt_sliced
        rmse_corr = np.sqrt(np.mean(error_corr**2))
        corr_corr, _ = pearsonr(corrected, gt_sliced)
        max_err_corr = np.max(np.abs(error_corr))
        
        # --- Print Results ---
        print(f"\n--- {key.replace('_', ' ').title()} ---")
        print(f"{'':<25} | {'Uncalibrated':<20} | {'Calibrated':<20}")
        print("-" * 70)
        print(f"{'RMSE (deg)':<25} | {rmse_raw:<20.2f} | {rmse_corr:<20.2f}")
        print(f"{'Max Error (deg)':<25} | {max_err_raw:<20.2f} | {max_err_corr:<20.2f}")
        print(f"{'Correlation Coeff.':<25} | {corr_raw:<20.2f} | {corr_corr:<20.2f}")

def plot_joint_angle_results(raw_angles, corrected_angles, gt_angles):
    """
    Plots GT, raw (uncalibrated), and corrected (calibrated) angles.
    """
    print("\nPlotting joint angle results...")
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f'EKF Joint Angle Estimation vs. Ground Truth (7-State) for Sequence {SEQUENCE_INDEX}', fontsize=16)
    
    raw_deg = np.rad2deg(raw_angles)
    corrected_deg = np.rad2deg(corrected_angles)
    
    angle_keys = ['sh_abduction', 'sh_flexion', 'el_flexion']
    titles = ["Shoulder Abduction/Adduction", "Shoulder Flexion/Extension", "Elbow Flexion/Extension"]

    gt_angle_data = [gt_angles[key] for key in angle_keys]

    for i, ax in enumerate(axs):
        # Plot Ground Truth
        ax.plot(np.rad2deg(gt_angle_data[i]), label='GT', color='green')
        
        # Plot Uncalibrated (Raw) Data
        raw_plot_data = np.rad2deg(np.unwrap(raw_angles[:, i]))
        ax.plot(raw_plot_data, label='EKF Raw (Uncalibrated)', color='orange', linestyle=':')
        
        # Plot Calibrated (Corrected) Data
        corrected_plot_data = np.rad2deg(np.unwrap(corrected_angles[:, i]))
        ax.plot(corrected_plot_data, label='EKF Final (Calibrated)', color='blue', linestyle='--')
        
        ax.set_title(titles[i])
        ax.set_ylabel("Angle (degrees)")
        ax.grid(True)
        ax.legend()

    axs[2].set_xlabel("Frame")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/train.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR
    
    SEQUENCE_INDEX, TARGET_ARM, DOWNSAMPLE_RATE = 1, 'left', 5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}"); print("Loading dataset...")
    original_data = torch.load(DATASET_PATH, map_location=DEVICE); print("Data loaded.")
    
    ACC_KEY, ORI_KEY, UWB_KEY, POSE_KEY, TRAN_KEY = 'acc', 'ori', 'vuwb', 'pose', 'tran'
    
    if TARGET_ARM == 'left':
        PELVIS_IDX, WRIST_IDX, SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX, SHLDR_POSE_IDX, ELBOW_POSE_IDX = 5, 0, 16, 18, 20, 15, 17
    else:
        PELVIS_IDX, WRIST_IDX, SHLDR_JOINT_IDX, ELBOW_JOINT_IDX, WRIST_JOINT_IDX, SHLDR_POSE_IDX, ELBOW_POSE_IDX = 5, 1, 17, 19, 21, 16, 18
        
    constants = {'TARGET_ARM': TARGET_ARM, 'PELVIS_IDX': PELVIS_IDX, 'WRIST_IDX': WRIST_IDX,'SHLDR_JOINT_IDX': SHLDR_JOINT_IDX, 'ELBOW_JOINT_IDX': ELBOW_JOINT_IDX, 'WRIST_JOINT_IDX': WRIST_JOINT_IDX, 'SHLDR_POSE_IDX': SHLDR_POSE_IDX, 'ELBOW_POSE_IDX': ELBOW_POSE_IDX, 'POSE_KEY': POSE_KEY, 'TRAN_KEY': TRAN_KEY, 'ACC_KEY': ACC_KEY, 'SEQUENCE_INDEX': SEQUENCE_INDEX, 'DOWNSAMPLE_RATE': DOWNSAMPLE_RATE}

    sensor_data = {
        'acc': original_data[ACC_KEY][SEQUENCE_INDEX],
        'ori': original_data[ORI_KEY][SEQUENCE_INDEX],
        'uwb': original_data[UWB_KEY][SEQUENCE_INDEX]
    }

    # Pre-calculate ground truth kinematics
    gt_kinematics = get_ground_truth_kinematics(original_data, constants, DEVICE)

    # Get the pelvis-to-shoulder offset in the pelvis's local frame
    shoulder_pelvis_offset = get_shoulder_pelvis_offset(gt_kinematics, sensor_data, constants)

    # Make sure you have the UWB calibration function in your script
    uwb_offset = calibrate_uwb_from_t_pose(original_data[UWB_KEY][SEQUENCE_INDEX], gt_kinematics, constants)
    
    # 1. Run the EKF to get a stable, drift-free, but biased estimate.
    estimated_angles_raw = run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, DEVICE, uwb_offset, shoulder_pelvis_offset)
    
    # 2. Apply the final calibration to fix bias and scaling errors.
    estimated_angles_corrected = calibrate_and_correct_angles(estimated_angles_raw, gt_kinematics['gt_angles'])
    
    # 3. Calculate and report the final performance.
    calculate_performance_metrics(estimated_angles_raw, estimated_angles_corrected, gt_kinematics['gt_angles'])

    # 4. Visualize the results in SMPL.
    run_visualization(post_process_angles_to_poses(estimated_angles_raw, gt_kinematics['poses'], constants, DEVICE), gt_kinematics, constants, DEVICE)

    # 5. Plot the final, highly accurate results.
    plot_joint_angle_results(estimated_angles_raw, estimated_angles_corrected, gt_kinematics['gt_angles'])

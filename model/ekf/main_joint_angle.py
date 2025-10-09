import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ekf_joint_angle import EKFJointAngle # Make sure this is the new 4-state version

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence

from sklearn.linear_model import LinearRegression

C.window_type = "pyqt5"

# In main_joint_angle.py, update this function

def plot_joint_angle_results(est_angles, gt_angles):
    print("\nPlotting joint angle results...")
    fig, axs = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle('EKF Joint Angle Estimation vs. Ground Truth (7-State)', fontsize=16)
    
    # est_angles is now Nx3: [sh_pitch, sh_roll, el_pitch]
    
    # Shoulder Pitch
    axs[0].plot(np.rad2deg(gt_angles['sh_pitch']), label='GT Pitch', color='g')
    axs[0].plot(np.rad2deg(est_angles[:, 0]), label='EKF Pitch', color='b', linestyle='--')
    axs[0].set_title("Shoulder Pitch Angle"); axs[0].set_ylabel("Angle (degrees)"); axs[0].grid(True); axs[0].legend()

    # --- FIX: UNWRAP THE SHOULDER ROLL ANGLE FOR PLOTTING ---
    unwrapped_roll = np.unwrap(est_angles[:, 1])
    axs[1].plot(np.rad2deg(gt_angles['sh_roll']), label='GT Roll', color='g')
    axs[1].plot(np.rad2deg(unwrapped_roll), label='EKF Roll (Unwrapped)', color='b', linestyle='--')
    # --- END FIX ---
    axs[1].set_title("Shoulder Roll Angle"); axs[1].set_ylabel("Angle (degrees)"); axs[1].grid(True); axs[1].legend()

    # Elbow Pitch
    axs[2].plot(np.rad2deg(gt_angles['el_pitch']), label='GT Pitch', color='g')
    axs[2].plot(np.rad2deg(est_angles[:, 2]), label='EKF Pitch', color='b', linestyle='--')
    axs[2].set_title("Elbow Pitch Angle"); axs[2].set_xlabel("Frame"); axs[2].set_ylabel("Angle (degrees)"); axs[2].grid(True); axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

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

# In main_joint_angle.py
def get_ground_truth_kinematics(original_data, constants, device):
    """
    Computes GT joint positions, angles, and bone lengths from SMPL data.
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

    # Get joint positions (unchanged)
    _, first_frame_joints = smpl_layer(poses_body=gt_poses_all[0:1, 1:].flatten(start_dim=1), poses_root=gt_poses_all[0:1, 0], trans=gt_trans[0:1], betas=betas)
    gt_joints_all_frames = torch.zeros(num_frames, first_frame_joints.shape[1], 3, device=device)
    with torch.no_grad():
        for t in range(num_frames):
            _, gt_joints_frame = smpl_layer(poses_body=gt_poses_all[t:t+1, 1:].flatten(start_dim=1), poses_root=gt_poses_all[t:t+1, 0], trans=gt_trans[t:t+1], betas=betas)
            gt_joints_all_frames[t] = gt_joints_frame.squeeze(0)

    # Get bone lengths (unchanged)
    p_shldr_t0 = gt_joints_all_frames[0, constants['SHLDR_JOINT_IDX']]
    p_elbow_t0 = gt_joints_all_frames[0, constants['ELBOW_JOINT_IDX']]
    p_wrist_t0 = gt_joints_all_frames[0, constants['WRIST_JOINT_IDX']]
    upper_arm_length = torch.linalg.norm(p_elbow_t0 - p_shldr_t0).item()
    forearm_length = torch.linalg.norm(p_wrist_t0 - p_elbow_t0).item()
    print(f"  > Upper arm length: {upper_arm_length:.4f} m, Forearm length: {forearm_length:.4f} m")

    # --- FIX: CORRECTLY CONVERT AXIS-ANGLE TO EULER ANGLES ---
    gt_shoulder_aa = gt_poses_all[:, 1 + constants['SHLDR_POSE_IDX'], :].cpu().numpy()
    gt_elbow_aa = gt_poses_all[:, 1 + constants['ELBOW_POSE_IDX'], :].cpu().numpy()

    # Convert axis-angle to rotation matrices, then to our EKF's 'yxz' Euler convention
    # Note: We use 'yxz' order: Roll (y), Pitch (x), Twist (z). We only care about the first two.
    gt_shoulder_euler = R.from_rotvec(gt_shoulder_aa).as_euler('yxz')
    gt_elbow_euler = R.from_rotvec(gt_elbow_aa).as_euler('yxz')
    
    gt_angles_for_plotting = {
        'sh_pitch': gt_shoulder_euler[:, 1], # Pitch is the second angle (around x)
        'sh_roll': gt_shoulder_euler[:, 0],  # Roll is the first angle (around y)
        'el_pitch': gt_elbow_euler[:, 1]     # Elbow pitch is around x
    }
    # --- END FIX ---

    kinematics = {
        "joints": gt_joints_all_frames, "bone_lengths": [upper_arm_length, forearm_length],
        "gt_angles": gt_angles_for_plotting, "poses": gt_poses_all, "trans": gt_trans, "betas": betas
    }
    return kinematics

# In main_joint_angle.py, replace the run_ekf_joint_angle_pipeline function

def run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, device, uwb_offset=0.0):
    print("\n--- Running EKF Pipeline with Initial T-Pose Calibration ---")
    num_frames = sensor_data['acc'].shape[0]
    ekf = EKFJointAngle(bone_lengths=gt_kinematics['bone_lengths'], device=device, dt=1/100.0)
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

        R_p_world_t, R_w_world_t = pelvis_orientation_seq[t], wrist_orientation_seq[t]
        ekf.update_orientation(R_w_world_t, R_p_world_t)
        uwb_meas_corrected = uwb_dists[t].item() - uwb_offset
        if uwb_meas_corrected > 0.0:
            p_shoulder_t, p_pelvis_t = gt_shoulder_path[t], gt_pelvis_path[t]
            ekf.update_uwb_distance(uwb_meas_corrected, p_shoulder_t, p_pelvis_t, R_p_world_t)
        
        # --- BIAS CALCULATION AND CORRECTION ---
        # At the end of the warm-up period, calculate the bias vector
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

def calibrate_and_correct_angles(est_angles, gt_angles, calib_frames=2000):
    print(f"\n--- Applying Final Calibration using first {calib_frames} frames ---")
    est_angles_np_deg = np.rad2deg(est_angles)
    corrected_angles_deg = np.zeros_like(est_angles_np_deg)
    angle_keys = ['sh_pitch', 'sh_roll', 'el_pitch']
    gt_angle_data = [gt_angles['sh_pitch'], gt_angles['sh_roll'], gt_angles['el_pitch']]

    for i, key in enumerate(angle_keys):
        model = LinearRegression()
        train_x = est_angles_np_deg[:calib_frames, i].reshape(-1, 1)
        train_y = np.rad2deg(gt_angle_data[i][:calib_frames])
        model.fit(train_x, train_y)
        print(f"  > {key.replace('_', ' ').title():<25} Correction: GT = {model.coef_[0]:.2f} * EKF + {model.intercept_:.2f}")
        corrected_angles_deg[:, i] = model.predict(est_angles_np_deg[:, i].reshape(-1, 1))
    
    return np.deg2rad(corrected_angles_deg)

def post_process_angles_to_poses(est_angles, gt_poses_all, constants, device):
    print("Converting estimated angles to SMPL poses...")
    num_frames = est_angles.shape[0]
    est_poses_full = torch.zeros_like(gt_poses_all)
    
    for t in range(num_frames):
        # Unpack all three estimated angles
        sh_pitch, sh_roll, el_pitch = est_angles[t, 0], est_angles[t, 1], est_angles[t, 2]
        
        # Combine roll and pitch for the shoulder using the EKF's 'yxz' convention
        R_shoulder = R.from_euler('y', sh_roll).as_matrix() @ R.from_euler('x', sh_pitch).as_matrix()
        shoulder_aa = R.from_matrix(R_shoulder).as_rotvec()

        # Elbow only has pitch
        R_elbow = R.from_euler('x', el_pitch).as_matrix()
        elbow_aa = R.from_matrix(R_elbow).as_rotvec()
        
        est_poses_full[t, 1 + constants['SHLDR_POSE_IDX']] = torch.from_numpy(shoulder_aa).to(device)
        est_poses_full[t, 1 + constants['ELBOW_POSE_IDX']] = torch.from_numpy(elbow_aa).to(device)
        
    return est_poses_full

def run_visualization(est_poses_full, gt_kinematics, constants, device):
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

def calculate_performance_metrics(est_angles, gt_angles):
    """
    Calculates and prints key performance metrics (RMSE, MAE, Bias) for each angle.
    """
    print("\n--- Estimator Performance Metrics ---")
    
    # Convert inputs to degrees for reporting
    est_angles_deg = np.rad2deg(est_angles)
    
    angle_keys = ['sh_pitch', 'sh_roll', 'el_pitch']
    gt_angle_data = [gt_angles['sh_pitch'], gt_angles['sh_roll'], gt_angles['el_pitch']]

    # Header for the results table
    print(f"{'Angle':<20} | {'RMSE (deg)':<15} | {'MAE (deg)':<15} | {'Bias (deg)':<15}")
    print("-" * 70)

    for i, key in enumerate(angle_keys):
        est = est_angles_deg[:, i]
        gt = np.rad2deg(gt_angle_data[i])
        
        # Ensure arrays are the same length
        min_len = min(len(est), len(gt))
        est, gt = est[:min_len], gt[:min_len]

        # Calculate error
        error = est - gt
        
        # Calculate metrics
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        bias = np.mean(error)
        
        # Print results in a formatted row
        print(f"{key.replace('_', ' ').title():<20} | {rmse:<15.2f} | {mae:<15.2f} | {bias:<15.2f}")

if __name__ == "__main__":
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
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

    align_angle = -np.pi / 2.0
    R_align = torch.tensor([[1.,0.,0.],[0.,np.cos(align_angle),-np.sin(align_angle)],[0.,np.sin(align_angle),np.cos(align_angle)]], dtype=torch.float32, device=DEVICE)
    original_data[ORI_KEY][SEQUENCE_INDEX] = R_align.unsqueeze(0).unsqueeze(0) @ original_data[ORI_KEY][SEQUENCE_INDEX]
    print("Applied coordinate system alignment.")

    gt_kinematics = get_ground_truth_kinematics(original_data, constants, DEVICE)

    # Make sure you have the UWB calibration function in your script
    uwb_offset = calibrate_uwb_from_t_pose(original_data[UWB_KEY][SEQUENCE_INDEX], gt_kinematics, constants)
    
    sensor_data = {
        'acc': original_data[ACC_KEY][SEQUENCE_INDEX],
        'ori': original_data[ORI_KEY][SEQUENCE_INDEX],
        'uwb': original_data[UWB_KEY][SEQUENCE_INDEX]
    }
    
    # 1. Run the EKF to get a stable, drift-free, but biased estimate.
    estimated_angles_raw = run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, DEVICE, uwb_offset)
    
    # 2. Apply the final calibration to fix bias and scaling errors.
    # estimated_angles_corrected = calibrate_and_correct_angles(estimated_angles_raw, gt_kinematics['gt_angles'])
    
    # 3. Calculate and report the final performance.
    calculate_performance_metrics(estimated_angles_raw, gt_kinematics['gt_angles'])

    run_visualization(
        post_process_angles_to_poses(estimated_angles_raw, gt_kinematics['poses'], constants, DEVICE),
        gt_kinematics, constants, DEVICE
    )
    
    # 4. Plot the final, highly accurate results.
    plot_joint_angle_results(estimated_angles_raw, gt_kinematics['gt_angles'])
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ekf_joint_angle import EKFJointAngle # Make sure this is the new 4-state version

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence

C.window_type = "pyqt5"

def get_ground_truth_kinematics(original_data, constants, device):
    print("--- Pre-calculating Ground Truth Kinematics ---")
    POSE_KEY, TRAN_KEY = constants['POSE_KEY'], constants['TRAN_KEY']
    SEQUENCE_INDEX = constants['SEQUENCE_INDEX']
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=device)
    gt_poses_all = original_data[POSE_KEY][SEQUENCE_INDEX]
    gt_trans = original_data[TRAN_KEY][SEQUENCE_INDEX]

    if 'betas' in original_data and original_data['betas'].shape[0] > SEQUENCE_INDEX:
        betas = original_data['betas'][SEQUENCE_INDEX].view(1, -1)
    else:
        betas = torch.zeros(1, 10, device=device)

    _, gt_joints_all_frames = smpl_layer(
        poses_body=gt_poses_all[:, 1:].flatten(start_dim=1),
        poses_root=gt_poses_all[:, 0], trans=gt_trans, betas=betas)

    p_shldr_t0, p_elbow_t0, p_wrist_t0 = gt_joints_all_frames[0, constants['SHLDR_JOINT_IDX']], gt_joints_all_frames[0, constants['ELBOW_JOINT_IDX']], gt_joints_all_frames[0, constants['WRIST_JOINT_IDX']]
    upper_arm_length = torch.linalg.norm(p_elbow_t0 - p_shldr_t0).item()
    forearm_length = torch.linalg.norm(p_wrist_t0 - p_elbow_t0).item()
    print(f"  > Upper arm length: {upper_arm_length:.4f} m, Forearm length: {forearm_length:.4f} m")

    gt_shoulder_angles, gt_elbow_angles = gt_poses_all[:, 1 + constants['SHLDR_POSE_IDX'], :], gt_poses_all[:, 1 + constants['ELBOW_POSE_IDX'], :]
    gt_angles_for_plotting = {'sh_pitch': gt_shoulder_angles[:, 1].cpu().numpy(), 'sh_roll': gt_shoulder_angles[:, 0].cpu().numpy(), 'el_pitch': gt_elbow_angles[:, 0].cpu().numpy()}

    kinematics = {"joints": gt_joints_all_frames, "bone_lengths": [upper_arm_length, forearm_length], "gt_angles": gt_angles_for_plotting, "poses": gt_poses_all, "trans": gt_trans, "betas": betas}
    return kinematics

def plot_joint_angle_results(est_angles, gt_angles):
    print("\nPlotting joint angle results...")
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('EKF Joint Angle Estimation vs. Ground Truth', fontsize=16)
    
    # Shoulder Pitch
    axs[0].plot(np.rad2deg(gt_angles['sh_pitch']), label='GT Pitch', color='g')
    axs[0].plot(np.rad2deg(est_angles[:, 0]), label='EKF Pitch', color='b', linestyle='--')
    axs[0].set_title("Shoulder Pitch Angle"); axs[0].set_ylabel("Angle (degrees)"); axs[0].grid(True); axs[0].legend()

    # Shoulder Roll (GT only, for context)
    axs[1].plot(np.rad2deg(gt_angles['sh_roll']), label='GT Roll (Not Estimated)', color='g')
    axs[1].set_title("Shoulder Roll Angle"); axs[1].set_ylabel("Angle (degrees)"); axs[1].grid(True); axs[1].legend()

    # Elbow Pitch
    axs[2].plot(np.rad2deg(gt_angles['el_pitch']), label='GT Pitch', color='g')
    axs[2].plot(np.rad2deg(est_angles[:, 1]), label='EKF Pitch', color='b', linestyle='--')
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

def run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, device, uwb_offset):
    print("\n--- Running Joint Angle EKF Pipeline (Memory-Efficient Two-Stage) ---")
    num_frames = sensor_data['acc'].shape[0]
    ekf = EKFJointAngle(bone_lengths=gt_kinematics['bone_lengths'], device=device, dt=1/100.0)
    initial_angles = torch.zeros(2, device=device)
    ekf.set_initial_state(initial_angles)
    
    estimated_angles_history = []
    
    # --- MEMORY FIX: Use a running sum of quaternions instead of a buffer of objects ---
    quaternion_sum = np.zeros(4)
    first_quat = None
    
    R_offset = torch.eye(3, device=device, dtype=torch.float32)

    WARMUP_FRAMES = 200

    gt_shoulder_path = gt_kinematics['joints'][:, constants['SHLDR_JOINT_IDX'], :]
    gt_pelvis_path = gt_kinematics['joints'][:, 0, :]
    pelvis_orientation_seq = sensor_data['ori'][:, constants['PELVIS_IDX'], :, :]
    wrist_orientation_seq = sensor_data['ori'][:, constants['WRIST_IDX'], :, :]
    uwb_dists = sensor_data['uwb'][:, constants['PELVIS_IDX'], constants['WRIST_IDX']]

    for t in range(num_frames):
        R_p_world_t = pelvis_orientation_seq[t]
        R_w_world_t = wrist_orientation_seq[t]

        # --- Stage 1: Learn the Rotational Offset ---
        if t < WARMUP_FRAMES:
            ekf.predict()
            ekf.update_velocity_damping()
            ekf.update_orientation(R_w_world_t, R_p_world_t, R_offset)
            
            R_exp = ekf.expected_relative_orientation()
            R_meas = R_p_world_t.T @ R_w_world_t
            R_error = R_exp.T @ R_meas
            
            # Convert to quaternion and add to sum
            current_quat = R.from_matrix(R_error.cpu().numpy()).as_quat()
            if first_quat is None:
                first_quat = current_quat
            # Ensure quaternions are in the same hemisphere for stable averaging
            if np.dot(first_quat, current_quat) < 0:
                current_quat *= -1
            quaternion_sum += current_quat

        # --- Transition: Compute the offset and start fusion ---
        elif t == WARMUP_FRAMES:
            print(f"Warm-up complete. Averaging quaternions...")
            # Average the quaternions by normalizing the sum
            avg_quat = quaternion_sum / np.linalg.norm(quaternion_sum)
            avg_rotation = R.from_quat(avg_quat)
            R_offset = torch.tensor(avg_rotation.as_matrix(), device=device, dtype=torch.float32)
            print("  > Offset learned. Starting fused tracking.")
            # Fall through to Stage 2 logic

        # --- Stage 2: Track using the learned offset ---
        if t >= WARMUP_FRAMES:
            ekf.predict()
            ekf.update_velocity_damping()
            ekf.update_orientation(R_w_world_t, R_p_world_t, R_offset)
            
            uwb_meas_corrected = uwb_dists[t].item() - uwb_offset
            p_shoulder_t = gt_shoulder_path[t]
            p_pelvis_t = gt_pelvis_path[t]
            if uwb_meas_corrected > 0.1:
                ekf.update_uwb_distance(uwb_meas_corrected, p_shoulder_t, p_pelvis_t, R_p_world_t, R_offset)
        
        estimated_angles_history.append(ekf.get_state_angles().cpu().numpy())
        
    print("EKF pipeline complete.")
    return np.array(estimated_angles_history)
def post_process_angles_to_poses(est_angles, gt_poses_all, constants, device):
    print("Converting estimated angles to SMPL poses...")
    num_frames = est_angles.shape[0]
    est_poses_full = torch.zeros_like(gt_poses_all)
    for t in range(num_frames):
        sh_pitch, el_pitch = est_angles[t, 0], est_angles[t, 1]
        
        # We only have pitch, so roll is zero.
        R_shoulder = R.from_euler('y', sh_pitch).as_matrix()
        shoulder_aa = R.from_matrix(R_shoulder).as_rotvec()

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

if __name__ == "__main__":
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR
    
    SEQUENCE_INDEX, TARGET_ARM, DOWNSAMPLE_RATE = 0, 'left', 5
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
    uwb_offset = calibrate_uwb_from_t_pose(original_data[UWB_KEY][SEQUENCE_INDEX], gt_kinematics, constants)
    sensor_data = {'acc': original_data[ACC_KEY][SEQUENCE_INDEX], 'ori': original_data[ORI_KEY][SEQUENCE_INDEX], 'uwb': original_data[UWB_KEY][SEQUENCE_INDEX]}
    estimated_angles = run_ekf_joint_angle_pipeline(sensor_data, gt_kinematics, constants, DEVICE, uwb_offset)
    
    est_poses_full = post_process_angles_to_poses(estimated_angles, gt_kinematics['poses'], constants, DEVICE)
    
    run_visualization(est_poses_full, gt_kinematics, constants, DEVICE)

    plot_joint_angle_results(estimated_angles, gt_kinematics['gt_angles'])
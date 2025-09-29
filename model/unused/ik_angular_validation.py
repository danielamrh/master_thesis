import torch
import smplx # We need this to get the ground truth joint positions
import os
from model.ekf.ekf_estimation import EKF # Assumes ekf_estimation.py is in the same folder

def validate_ekf_positions(original_data, sequence_index):
    """
    Runs the EKF and compares its estimated 3D joint positions against the
    ground truth from the SMPL model.
    """
    print(f"--- Running Positional Validation for EKF on Sequence {sequence_index} ---")
    
    # --- Configuration ---
    ACC_KEY, ORI_KEY, UWB_KEY = 'acc', 'ori', 'vuwb'
    POSE_KEY, TRAN_KEY = 'pose', 'tran'
    PELVIS_IDX, L_WRIST_IDX_SENSOR = 5, 0 # Sensor indices
    
    # SMPL joint indices are different from sensor indices
    SMPL_PELVIS_IDX, SMPL_L_WRIST_IDX = 0, 20
    # ---------------------

    # --- Run the EKF Pipeline (as before) ---
    acc_local_seq = original_data[ACC_KEY][sequence_index]
    ori_matrix_seq = original_data[ORI_KEY][sequence_index]
    uwb_dist_matrix = original_data[UWB_KEY][sequence_index]
    num_frames = acc_local_seq.shape[0]
    
    ekf = EKF()
    initial_dist = uwb_dist_matrix[0, PELVIS_IDX, L_WRIST_IDX_SENSOR].item()
    ekf.set_initial(
        p_p_init=torch.tensor([0., 0., 1.2]), v_p_init=torch.zeros(3),
        p_w_init=torch.tensor([-initial_dist, 0.0, 1.2]), v_w_init=torch.zeros(3)
    )
    
    all_est_states = []
    for t in range(num_frames):
        R_p, R_w = ori_matrix_seq[t, PELVIS_IDX], ori_matrix_seq[t, L_WRIST_IDX_SENSOR]
        acc_p_world = (R_p @ acc_local_seq[t, PELVIS_IDX].unsqueeze(1)).squeeze()
        acc_w_world = (R_w @ acc_local_seq[t, L_WRIST_IDX_SENSOR].unsqueeze(1)).squeeze()
        ekf.predict(acc_p_world, acc_w_world)
        dist_meas = uwb_dist_matrix[t, PELVIS_IDX, L_WRIST_IDX_SENSOR].item()
        if 0.2 < dist_meas < 2.0: ekf.update_uwb_scalar_distance(dist_meas)
        ekf.update_zero_velocity_pelvis()
        state, _ = ekf.get_state()
        all_est_states.append(state)
    est_states = torch.stack(all_est_states)
    
    # --- Calculate Ground Truth Positions ---
    # The dataset's `pose` and `tran` are inputs to the SMPL model, which then gives us the 3D joint locations.
    smpl_model = smplx.create(os.environ["SMPLX_MODEL_DIR"], model_type='smpl', gender='male')
    gt_poses_all = original_data[POSE_KEY][sequence_index]
    gt_trans = original_data[TRAN_KEY][sequence_index]
    
    smpl_output_gt = smpl_model(
        global_orient=gt_poses_all[:, 0, :],
        body_pose=gt_poses_all[:, 1:24, :].flatten(start_dim=1),
        transl=gt_trans
    )
    gt_joints_3d = smpl_output_gt.joints

    # --- Compare and Calculate Error ---
    pelvis_errors = []
    wrist_errors = []
    
    for t in range(num_frames):
        # Get our EKF's estimated positions for this frame
        est_pelvis_pos = est_states[t].squeeze()[0:3]
        est_wrist_pos = est_states[t].squeeze()[6:9]
        
        # Get the ground truth positions for this frame
        gt_pelvis_pos = gt_joints_3d[t, SMPL_PELVIS_IDX, :]
        gt_wrist_pos = gt_joints_3d[t, SMPL_L_WRIST_IDX, :]
        
        # Calculate Euclidean distance error
        pelvis_errors.append(torch.linalg.norm(est_pelvis_pos - gt_pelvis_pos).item())
        wrist_errors.append(torch.linalg.norm(est_wrist_pos - gt_wrist_pos).item())

    # --- Print Results ---
    mean_pelvis_error = (sum(pelvis_errors) / len(pelvis_errors)) * 100 # convert to cm
    mean_wrist_error = (sum(wrist_errors) / len(wrist_errors)) * 100 # convert to cm
    
    print("\n--- EKF Position Validation Results ---")
    print(f"Mean Pelvis Position Error: {mean_pelvis_error:.2f} cm")
    print(f"Mean Wrist Position Error:  {mean_wrist_error:.2f} cm")
    print("---------------------------------------")


if __name__ == "__main__":
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR

    original_data = torch.load(DATASET_PATH, map_location="cpu")
    
    # Validate a sequence (e.g., Sequence 3)
    validate_ekf_positions(original_data, sequence_index=3)

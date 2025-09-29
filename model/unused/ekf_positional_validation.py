import torch
import smplx
import os
from model.ekf.ekf_estimation import EKF

def validate_ekf_positions(original_data, sequence_index, target_arm='left'):
    """
    Runs the advanced EKF and compares its estimated 3D joint positions
    against the ground truth.
    """
    print(f"--- Running Advanced EKF Validation on Sequence {sequence_index} ({target_arm.title()} Arm) ---")
    
    ACC_KEY, ORI_KEY, UWB_KEY, POSE_KEY, TRAN_KEY = 'acc', 'ori', 'vuwb', 'pose', 'tran'
    
    # Using your specified index mapping
    if target_arm == 'left':
        PELVIS_IDX, WRIST_IDX_SENSOR = 5, 0
        SMPL_WRIST_IDX = 19
        ik_ref_vec = torch.tensor([1.0, 0.0, 0.0])
    else: # right
        PELVIS_IDX, WRIST_IDX_SENSOR = 5, 1
        SMPL_WRIST_IDX = 20
        ik_ref_vec = torch.tensor([-1.0, 0.0, 0.0])
    SMPL_PELVIS_IDX = 2

    acc_local_seq = original_data[ACC_KEY][sequence_index]
    ori_matrix_seq = original_data[ORI_KEY][sequence_index]
    uwb_dist_matrix = original_data[UWB_KEY][sequence_index]
    num_frames = acc_local_seq.shape[0]
    
    ekf = EKF()
    initial_dist = uwb_dist_matrix[0, PELVIS_IDX, WRIST_IDX_SENSOR].item()
    ekf.set_initial(
        p_p_init=torch.tensor([0., 0., 1.2]), v_p_init=torch.zeros(3),
        p_w_init=torch.tensor([ik_ref_vec[0] * initial_dist, 0.0, 1.2]), v_w_init=torch.zeros(3)
    )
    
    all_est_states = []
    for t in range(num_frames):
        acc_p_local = acc_local_seq[t, PELVIS_IDX]
        acc_w_local = acc_local_seq[t, WRIST_IDX_SENSOR]
        R_p, R_w = ori_matrix_seq[t, PELVIS_IDX], ori_matrix_seq[t, WRIST_IDX_SENSOR]
        
        acc_p_world = (R_p @ acc_p_local.unsqueeze(1)).squeeze()
        acc_w_world = (R_w @ acc_w_local.unsqueeze(1)).squeeze()
        
        ekf.predict(acc_p_world, acc_w_world)
        
        dist_meas = uwb_dist_matrix[t, PELVIS_IDX, WRIST_IDX_SENSOR].item()
        if 0.2 < dist_meas < 2.0:
            ekf.update_uwb_scalar_distance(dist_meas)

        # Pass the local acceleration to the ZUPT detector
        ekf.update_zero_velocity_pelvis(acc_p_local)
        
        state, _ = ekf.get_state()
        all_est_states.append(state)
    est_states = torch.stack(all_est_states)
    
    smpl_model = smplx.create(os.environ["SMPLX_MODEL_DIR"], model_type='smpl', gender='male')
    gt_poses_all = original_data[POSE_KEY][sequence_index]
    gt_trans = original_data[TRAN_KEY][sequence_index]
    
    pelvis_errors = []
    wrist_errors = []
    
    print("Calculating ground truth positions and comparing frame by frame...")
    for t in range(num_frames):
        smpl_output_gt = smpl_model(
            global_orient=gt_poses_all[t:t+1, 0, :],
            body_pose=gt_poses_all[t:t+1, 1:24, :].flatten(start_dim=1),
            transl=gt_trans[t:t+1]
        )
        gt_joints_3d = smpl_output_gt.joints
        
        est_pelvis_pos = est_states[t].squeeze()[0:3]
        est_wrist_pos = est_states[t].squeeze()[6:9]
        
        gt_pelvis_pos = gt_joints_3d[0, SMPL_PELVIS_IDX, :]
        gt_wrist_pos = gt_joints_3d[0, SMPL_WRIST_IDX, :]
        
        pelvis_errors.append(torch.linalg.norm(est_pelvis_pos - gt_pelvis_pos).item())
        wrist_errors.append(torch.linalg.norm(est_wrist_pos - gt_wrist_pos).item())

    mean_pelvis_error = (sum(pelvis_errors) / len(pelvis_errors)) * 100
    mean_wrist_error = (sum(wrist_errors) / len(wrist_errors)) * 100
    
    print("\n--- EKF Position Validation Results ---")
    print(f"Mean Pelvis Position Error: {mean_pelvis_error:.2f} cm")
    print(f"Mean Wrist Position Error:  {mean_wrist_error:.2f} cm")
    print("---------------------------------------")

if __name__ == "__main__":
    DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"
    os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR

    original_data = torch.load(DATASET_PATH, map_location="cpu")
    
    validate_ekf_positions(original_data, sequence_index=0, target_arm='left')
    validate_ekf_positions(original_data, sequence_index=0, target_arm='right')


import torch
from model.ekf.ekf_estimation import EKF
from model.ekf.ik_solver_ekf import solve_fabrik, get_joint_angles_from_positions
import math

def run_motion_pipeline(original_data, sequence_index, dt=0.01, verbose=False):
    """
    Full pipeline: Runs the EKF and then solves for joint angles using FABRIK IK,
    with periodic status updates.
    """
    
    # --- Configuration ---
    ACC_KEY, ORI_KEY, UWB_KEY = 'acc', 'ori', 'vuwb' 
    PELVIS_IDX, WRIST_IDX = 5, 0
    # Set how often to print a status update (e.g., every 2000 frames)
    PRINT_INTERVAL = 2000
    # ---------------------

    try:
        acc_local_seq = original_data[ACC_KEY][sequence_index].detach().clone().float()
        ori_matrix_seq = original_data[ORI_KEY][sequence_index].detach().clone().float()
        uwb_dist_matrix = original_data[UWB_KEY][sequence_index].detach().clone().float()
        print("Successfully loaded all data from the original file.")
    except Exception as e:
        print(f"FATAL ERROR: Could not access required data. Error: {e}")
        return None, None

    num_frames = acc_local_seq.shape[0]
    acc_p_local = acc_local_seq[:, PELVIS_IDX]
    acc_w_local = acc_local_seq[:, WRIST_IDX]
    ori_p_matrix = ori_matrix_seq[:, PELVIS_IDX]
    ori_w_matrix = ori_matrix_seq[:, WRIST_IDX]
    
    ekf = EKF(dt=dt)
    initial_arm_length = uwb_dist_matrix[0, PELVIS_IDX, WRIST_IDX].item()
    if torch.isnan(torch.tensor(initial_arm_length)): initial_arm_length = 0.9
    ekf.set_initial(
        p_p_init=torch.tensor([0.0, 0.0, 1.2]),
        v_p_init=torch.zeros(3),
        p_w_init=torch.tensor([-initial_arm_length, 0.0, 1.2]),
        v_w_init=torch.zeros(3)
    )

    print("Running EKF and IK pipeline...")
    all_states = []
    all_joint_angles = []

    # Initialize the IK chain
    l1 = initial_arm_length * 0.55
    l2 = initial_arm_length * 0.45
    bone_lengths = torch.tensor([l1, l2])
    spine_pos_init = ekf.get_state()[0].squeeze()[0:3]
    shoulder_pos = spine_pos_init + torch.tensor([0.0, -0.15, -0.1])
    elbow_pos = shoulder_pos + torch.tensor([-l1, 0.0, 0.0])
    wrist_pos = elbow_pos + torch.tensor([-l2, 0.0, 0.0])
    current_chain_points = torch.stack([shoulder_pos, elbow_pos, wrist_pos])

    for t in range(num_frames):
        # --- EKF Prediction and Update ---
        R_p = ori_p_matrix[t]
        R_w = ori_w_matrix[t]
        acc_p_world = (R_p @ acc_p_local[t].unsqueeze(1)).squeeze()
        acc_w_world = (R_w @ acc_w_local[t].unsqueeze(1)).squeeze()
        ekf.predict(acc_p_world, acc_w_world)
        try:
            dist_meas = uwb_dist_matrix[t, PELVIS_IDX, WRIST_IDX].item()
            if not torch.isnan(torch.tensor(dist_meas)) and 0.2 < dist_meas < 2.0:
                ekf.update_uwb_scalar_distance(dist_meas)
        except Exception: pass
        ekf.update_zero_velocity_pelvis()
        
        # --- Log EKF State ---
        x, _ = ekf.get_state()
        all_states.append(x.clone() if not torch.isnan(x).any() else None)

        # --- IK Solving ---
        if x is not None:
            state = x.squeeze()
            target_wrist_pos = state[6:9]
            solved_chain_points = solve_fabrik(current_chain_points, target_wrist_pos, bone_lengths)
            joint_angles = get_joint_angles_from_positions(solved_chain_points)
            all_joint_angles.append(joint_angles)
            current_chain_points = solved_chain_points
        else:
            all_joint_angles.append(all_joint_angles[-1] if all_joint_angles else {"elbow_angle_rad": 0})

        # --- THE ADDED SECTION: Periodic Status Print ---
        if (t + 1) % PRINT_INTERVAL == 0:
            current_pelvis_pos = x.squeeze()[0:3]
            current_wrist_pos = x.squeeze()[6:9]
            current_dist = torch.linalg.norm(current_wrist_pos - current_pelvis_pos)
            current_elbow_angle = math.degrees(all_joint_angles[-1]['elbow_angle_rad'])
            
            print(f"\n--- Status at Frame {t+1}/{num_frames} ---")
            print(f"  EKF Pelvis (Spine) Pos: [{current_pelvis_pos[0]:.2f}, {current_pelvis_pos[1]:.2f}, {current_pelvis_pos[2]:.2f}]")
            print(f"  EKF Wrist Pos:          [{current_wrist_pos[0]:.2f}, {current_wrist_pos[1]:.2f}, {current_wrist_pos[2]:.2f}]")
            print(f"  EKF Distance:           {current_dist:.2f} m")
            print(f"  IK Elbow Angle:         {current_elbow_angle:.1f} degrees")
            print("--------------------------------------")
        # -----------------------------------------------

    est_states = torch.stack([s for s in all_states if s is not None])
    
    return est_states, all_joint_angles


if __name__ == "__main__":
    ORIGINAL_DATA_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
    SEQUENCE_INDEX = 3

    try:
        original_data = torch.load(ORIGINAL_DATA_PATH, map_location="cpu")
        
        est_states, joint_angles = run_motion_pipeline(original_data, SEQUENCE_INDEX)
        
        if est_states is not None:
            print("\n--- Pipeline Run Complete ---")
            
            # Final summary print remains useful
            final_ekf_state = est_states[-1].squeeze()
            final_pelvis_pos = final_ekf_state[0:3]
            final_wrist_pos = final_ekf_state[6:9]
            final_dist = torch.linalg.norm(final_wrist_pos - final_pelvis_pos)
            final_elbow_angle_deg = math.degrees(joint_angles[-1]['elbow_angle_rad'])

            print(f"Final Pelvis (Spine) Position: [{final_pelvis_pos[0]:.2f}, {final_pelvis_pos[1]:.2f}, {final_pelvis_pos[2]:.2f}]")
            print(f"Final Wrist Position:          [{final_wrist_pos[0]:.2f}, {final_wrist_pos[1]:.2f}, {final_wrist_pos[2]:.2f}]")
            print(f"Final EKF Distance:            {final_dist:.2f} m")
            print(f"Final IK Elbow Angle:          {final_elbow_angle_deg:.1f} degrees")
            
    except Exception as e:
        print(f"An error occurred: {e}")

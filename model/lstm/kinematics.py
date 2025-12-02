import torch
from config_amass import *
from utils import axis_angle_to_rotation_matrix 


def forward_kinematics(angles_batch, upper_arm_lengths, forearm_lengths):
    """
    Calculates the 3D positions of the elbow and wrist from the 6D axis-angle predictions.
    This function is fully differentiable.
    
    Args:
        angles_batch (torch.Tensor): Tensor of shape [batch_size, 6]
        upper_arm_lengths (torch.Tensor): Tensor of shape [batch_size]
        forearm_lengths (torch.Tensor): Tensor of shape [batch_size]
                                     
    Returns:
        torch.Tensor: Tensor of shape [batch_size, 6] (Elbow_xyz, Wrist_xyz).
    """
    batch_size = angles_batch.shape[0]
    device = angles_batch.device # Use device from input
    dtype = angles_batch.dtype

    # --- 1. Separate angles and convert to 3x3 rotation matrices ---
    shldr_angles = angles_batch[:, 0:3] # [batch, 3]
    elbow_angles = angles_batch[:, 3:6] # [batch, 3]

    R_shldr = axis_angle_to_rotation_matrix(shldr_angles) 
    R_elbow = axis_angle_to_rotation_matrix(elbow_angles) 

    # --- 2. Define "rest pose" bone vectors ---
    # Base vector is [-1, 0, 0], shape [1, 3, 1]
    v_rest_base = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3, 1)
    
    # Scale base vector by the batch of lengths
    # upper_arm_lengths is [B], needs to be [B, 1, 1]
    v_upper_arm = v_rest_base * upper_arm_lengths.view(batch_size, 1, 1).to(device=device, dtype=dtype)
    v_forearm = v_rest_base * forearm_lengths.view(batch_size, 1, 1).to(device=device, dtype=dtype)

    # --- 3. Apply rotations to get 3D positions (relative to shoulder) ---
    pos_elbow = torch.matmul(R_shldr, v_upper_arm) 
    
    R_full_arm = torch.matmul(R_shldr, R_elbow)
    pos_wrist = pos_elbow + torch.matmul(R_full_arm, v_forearm) 

    # --- 4. Concatenate results and return ---
    return torch.cat([pos_elbow.squeeze(-1), pos_wrist.squeeze(-1)], dim=1)


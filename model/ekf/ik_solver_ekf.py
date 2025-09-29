import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

def solve_fabrik(chain_points, target_pos, bone_lengths, tolerance=1e-3, max_iterations=15):
    """
    Solves a 3-point kinematic chain using the FABRIK algorithm.
    This finds the 3D position of the elbow joint.
    """
    num_joints = chain_points.shape[0]
    total_length = torch.sum(bone_lengths)
    base_pos = chain_points[0].clone()
    dist_to_target = torch.linalg.norm(target_pos - base_pos)

    # Handle unreachable target
    if dist_to_target >= total_length:
        direction = (target_pos - base_pos) / dist_to_target.clamp(min=1e-8)
        p_elbow = base_pos + direction * bone_lengths[0]
        p_wrist = p_elbow + direction * bone_lengths[1]
        return torch.stack([base_pos, p_elbow, p_wrist])

    # Standard FABRIK iterations
    points = chain_points.clone()
    for _ in range(max_iterations):
        if torch.linalg.norm(points[-1] - target_pos) < tolerance:
            break
        
        points[-1] = target_pos
        for i in range(num_joints - 2, -1, -1):
            dist = torch.linalg.norm(points[i+1] - points[i])
            direction = (points[i] - points[i+1]) / dist.clamp(min=1e-8)
            points[i] = points[i+1] + direction * bone_lengths[i]

        points[0] = base_pos
        for i in range(1, num_joints):
            dist = torch.linalg.norm(points[i] - points[i-1])
            direction = (points[i] - points[i-1]) / dist.clamp(min=1e-8)
            points[i] = points[i-1] + direction * bone_lengths[i-1]
            
    return points

def vec_to_rot(v_from, v_to):
    """
    Computes the rotation matrix that rotates v_from to v_to.
    """
    v_from = v_from / torch.linalg.norm(v_from).clamp(min=1e-8)
    v_to = v_to / torch.linalg.norm(v_to).clamp(min=1e-8)
    
    dot_product = torch.dot(v_from, v_to)
    
    if torch.abs(dot_product - 1.0) < 1e-6: # Same direction
        return torch.eye(3, device=v_from.device, dtype=v_from.dtype)
    if torch.abs(dot_product + 1.0) < 1e-6: # Opposite direction
        # Find an arbitrary perpendicular axis
        if torch.abs(v_from[0]) > 0.1:
            perp_axis = torch.tensor([0.0, 1.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        else:
            perp_axis = torch.tensor([1.0, 0.0, 0.0], device=v_from.device, dtype=v_from.dtype)
        axis = torch.cross(v_from, perp_axis)
        axis = axis / torch.linalg.norm(axis).clamp(min=1e-8)
        angle = np.pi
    else:
        axis = torch.cross(v_from, v_to)
        axis = axis / torch.linalg.norm(axis).clamp(min=1e-8)
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))

    # Rodrigues' rotation formula
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], device=v_from.device, dtype=v_from.dtype)
    
    R = torch.eye(3, device=v_from.device, dtype=v_from.dtype) + \
        torch.sin(angle) * K + \
        (1 - torch.cos(angle)) * (K @ K)
        
    return R
def positions_to_arm_rotations(p_pelvis, R_pelvis, p_shoulder, p_elbow, p_wrist, side='left'):
    """
    DEFINITIVE conversion from 3D joint positions to local SMPL axis-angle rotations
    using correct anatomical T-pose references. Correctly handles left/right mirroring.
    """
    # --- 1. Calculate Shoulder Rotation (relative to pelvis) ---
    upper_arm_world = p_elbow - p_shoulder
    R_pelvis_inv = torch.inverse(R_pelvis)
    upper_arm_local = R_pelvis_inv @ upper_arm_world

    # --- Handle left/right mirroring correctly ---
    if side == 'left':
        ref_vec_shoulder = torch.tensor([-1.0, 0.0, 0.0], device=p_pelvis.device, dtype=p_pelvis.dtype)
    else: # right
        ref_vec_shoulder = torch.tensor([-1.0, 0.0, 0.0], device=p_pelvis.device, dtype=p_pelvis.dtype)
        
    R_shoulder_local = vec_to_rot(ref_vec_shoulder, upper_arm_local)
    shoulder_aa = R.from_matrix(R_shoulder_local.cpu().detach().numpy()).as_rotvec()
    shoulder_aa = torch.from_numpy(shoulder_aa).to(p_pelvis.device, dtype=p_pelvis.dtype)

    # --- 2. Calculate Elbow Rotation (relative to shoulder) ---
    R_shoulder_world = R_pelvis @ R_shoulder_local
    forearm_world = p_wrist - p_elbow
    
    R_shoulder_world_inv = torch.inverse(R_shoulder_world)
    forearm_local = R_shoulder_world_inv @ forearm_world
    
    R_elbow_local = vec_to_rot(ref_vec_shoulder, forearm_local)
    elbow_aa = R.from_matrix(R_elbow_local.cpu().detach().numpy()).as_rotvec()
    elbow_aa = torch.from_numpy(elbow_aa).to(p_pelvis.device, dtype=p_pelvis.dtype)

    return shoulder_aa, elbow_aa


import torch
from torch.utils.data import Dataset
import numpy as np
from config import *

def rotation_matrix_to_axis_angle(R):
    """
    Convert a batch of rotation matrices to axis-angle vectors.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    safe_sin = torch.sin(angle).clamp(min=1e-5)
    axis_unnormalized = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    axis = axis_unnormalized / (2 * safe_sin.unsqueeze(-1))
    return axis * angle.unsqueeze(-1)


class UIPArmPoseDataset(Dataset):
    """
    Custom PyTorch Dataset for the Ultra Inertial Poser dataset.
    
    This version includes:
    1. 13-feature input (relative axis-angle orientation).
    2. Normalization for both inputs AND targets (poses).
    3. The nested stats dictionary structure required by evaluate.py.
    """
    def __init__(self, data_path, sequence_length, is_train=True, stats=None):
        print(f"Loading data from {data_path}...")
        self.sequence_length = sequence_length
        raw_data = torch.load(data_path, map_location='cpu')
        
        all_inputs = []
        all_targets = []

        num_sequences = len(raw_data[ACC_KEY])
        print(f"Found {num_sequences} sequences in the file.")

        for i in range(num_sequences):
            # --- Get Raw Sensor Data ---
            pelvis_acc = raw_data[ACC_KEY][i][:, PELVIS_IDX, :]
            wrist_acc = raw_data[ACC_KEY][i][:, WRIST_IDX, :]
            
            pelvis_ori_mat = raw_data[ORI_KEY][i][:, PELVIS_IDX, :, :]
            wrist_ori_mat = raw_data[ORI_KEY][i][:, WRIST_IDX, :, :]

            # --- Calculate Relative Orientation ---
            pelvis_ori_mat_transpose = torch.transpose(pelvis_ori_mat, 1, 2)
            relative_ori_mat = torch.bmm(pelvis_ori_mat_transpose, wrist_ori_mat)

            pelvis_ori_flat = pelvis_ori_mat.view(-1, 9)
            wrist_ori_flat = wrist_ori_mat.view(-1, 9)
            # wrist_ori_flat = relative_ori_mat.view(-1, 9)

            # --- Convert to Axis-Angle ---
            pelvis_ori_axis_angle = rotation_matrix_to_axis_angle(pelvis_ori_mat)
            relative_ori_axis_angle = rotation_matrix_to_axis_angle(relative_ori_mat)

            # --- Get UWB Distance ---
            uwb_dist = raw_data[UWB_KEY][i][:, PELVIS_IDX, WRIST_IDX].unsqueeze(-1)

            # --- Combine to 13 Features ---
            # input_features = torch.cat([
            #     pelvis_acc, pelvis_ori_axis_angle,
            #     wrist_acc, relative_ori_axis_angle,
            #     uwb_dist
            # ], dim=1)

            input_features = torch.cat([
                pelvis_acc, pelvis_ori_flat,
                wrist_acc, wrist_ori_flat,
                uwb_dist
            ], dim=1)

            # --- Target Data ---
            shldr_pose = raw_data[POSE_KEY][i][:, SHLDR_POSE_IDX, :] # 3D joint angles
            elbow_pose = raw_data[POSE_KEY][i][:, ELBOW_POSE_IDX, :] # 3D joint angles
            target_features = torch.cat([shldr_pose, elbow_pose], dim=1)
            
            all_inputs.append(input_features[::DOWNSAMPLE_RATE])
            all_targets.append(target_features[::DOWNSAMPLE_RATE])
            
        self.inputs = torch.cat(all_inputs, dim=0) 
        self.targets = torch.cat(all_targets, dim=0)
        
        print(f"Total frames after concatenation & downsampling: {self.inputs.shape[0]}")
        print(f"Input feature dimension: {self.inputs.shape[1]}") # Should print 13

        # --- Normalization Logic ---
        if is_train:
            self.mean = self.inputs.mean(dim=0)
            self.std = self.inputs.std(dim=0)
            self.std[self.std == 0] = 1.0
            self.target_mean = self.targets.mean(dim=0)
            self.target_std = self.targets.std(dim=0)
            self.target_std[self.target_std == 0] = 1.0
        else:
            if stats is None:
                raise ValueError("Validation/Test dataset requires stats from training set.")
            self.mean = stats['input_mean_std']['mean']
            self.std = stats['input_mean_std']['std']
            self.target_mean = stats['target_mean_std']['mean']
            self.target_std = stats['target_mean_std']['std']
            
        self.inputs = (self.inputs - self.mean) / self.std
        self.targets = (self.targets - self.target_mean) / self.target_std

    def get_stats(self):
        return {
            'input_mean_std': {'mean': self.mean, 'std': self.std}, # Normalization stats for inputs
            'target_mean_std': {'mean': self.target_mean, 'std': self.target_std} # Normalization stats for targets
        }

    def __len__(self):
        return self.inputs.shape[0] - self.sequence_length # Total available sequences

    def __getitem__(self, index):
        start = index # Starting index of the sequence
        end = index + self.sequence_length # Ending index (exclusive)
        input_seq = self.inputs[start:end, :] # Shape: (sequence_length, input_size)
        target_seq = self.targets[start:end, :] # Shape: (sequence_length, output_size) 
        return input_seq, target_seq


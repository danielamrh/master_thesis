import torch
from torch.utils.data import Dataset
import numpy as np
from config import *


class UIPArmPoseDataset(Dataset):
    """
    Custom PyTorch Dataset for the Ultra Inertial Poser dataset.
    This version flattens the 3x3 orientation matrices into 9D vectors.
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
            pelvis_acc = raw_data[ACC_KEY][i][:, PELVIS_IDX, :]
            wrist_acc = raw_data[ACC_KEY][i][:, WRIST_IDX, :]
            
            pelvis_ori_mat = raw_data[ORI_KEY][i][:, PELVIS_IDX, :, :]
            wrist_ori_mat = raw_data[ORI_KEY][i][:, WRIST_IDX, :, :]

            pelvis_ori_flat = pelvis_ori_mat.reshape(-1, 9)
            wrist_ori_flat = wrist_ori_mat.reshape(-1, 9)

            uwb_dist = raw_data[UWB_KEY][i][:, PELVIS_IDX, WRIST_IDX].unsqueeze(-1)

            # Combine all input features for a total of 25 features
            input_features = torch.cat([
                pelvis_acc, pelvis_ori_flat,
                wrist_acc, wrist_ori_flat,
                uwb_dist
            ], dim=1)

            # Target data remains the same
            shldr_pose = raw_data[POSE_KEY][i][:, SHLDR_POSE_IDX, :]
            elbow_pose = raw_data[POSE_KEY][i][:, ELBOW_POSE_IDX, :]
            target_features = torch.cat([shldr_pose, elbow_pose], dim=1)
            
            all_inputs.append(input_features[::DOWNSAMPLE_RATE])
            all_targets.append(target_features[::DOWNSAMPLE_RATE])
            
        self.inputs = torch.cat(all_inputs, dim=0)
        self.targets = torch.cat(all_targets, dim=0)
        
        print(f"Total frames after concatenation & downsampling: {self.inputs.shape[0]}")
        print(f"Input feature dimension: {self.inputs.shape[1]}") # Should print 25

        # Normalization logic remains the same
        if is_train:
            self.mean = self.inputs.mean(dim=0)
            self.std = self.inputs.std(dim=0)
            self.std[self.std == 0] = 1.0
        else:
            if stats is None:
                raise ValueError("Validation/Test dataset requires stats from training set.")
            self.mean = stats['mean']
            self.std = stats['std']
            
        self.inputs = (self.inputs - self.mean) / self.std

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def __len__(self):
        return self.inputs.shape[0] - self.sequence_length

    def __getitem__(self, index):
        start = index
        end = index + self.sequence_length
        
        input_seq = self.inputs[start:end, :]
        target_seq = self.targets[start:end, :]
        
        return input_seq, target_seq

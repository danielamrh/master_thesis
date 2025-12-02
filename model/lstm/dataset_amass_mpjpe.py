import torch
from torch.utils.data import Dataset
import numpy as np
from config_amass import * 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import os
import glob
from tqdm import tqdm
from utils import rotation_matrix_to_axis_angle

class UIPArmPoseDataset(Dataset):
    """
    Dataset class that handles both:
    1. AMASS directory structure (folders for 'pose', 'vacc', etc.)
    2. UIP single .pt file format
    """
    def __init__(self, data_path, sequence_length, sequence_indices, stride,
                 is_train=True, stats=None, create_windows=True,
                 augment=False, 
                 accel_noise_std=ACCEL_NOISE_STD, 
                 orient_noise_std=ORIENT_NOISE_STD, 
                 dist_noise_std=DIST_NOISE_STD,
                 scale_range=SCALE_RANGE, 
                 accel_offset_std=ACCEL_OFFSET_STD):
        
        self.num_windows = 0 
        print(f"Initializing dataset... Path: {data_path}")
        self.sequence_length = sequence_length
        self.stride = stride
        self.create_windows = create_windows
        
        self.augment = augment and is_train
        # Use config defined values
        self.accel_noise_std = accel_noise_std 
        self.orient_noise_std = orient_noise_std
        self.dist_noise_std = dist_noise_std
        self.scale_range = scale_range
        self.accel_offset_std = accel_offset_std
        
        raw_data = None

        # --- CASE 1: Single .pt File (UIP Data) ---
        if os.path.isfile(data_path):
            print(f"Detected single data file. Loading directly from {data_path}...")
            try:
                loaded_data = torch.load(data_path, map_location='cpu')
                if isinstance(loaded_data, dict):
                    raw_data = loaded_data
                else:
                    raw_data = loaded_data
                print("File loaded successfully.")
                
                # Heuristic: If specific keys are missing, try to map them
                self.acc_key = ACC_KEY if ACC_KEY in raw_data else 'acc'
                self.ori_key = ORI_KEY if ORI_KEY in raw_data else 'ori'
                self.uwb_key = UWB_KEY if UWB_KEY in raw_data else 'vuwb'
                self.pose_key = POSE_KEY if POSE_KEY in raw_data else 'pose'
                
                print(f"Using keys -> Acc: '{self.acc_key}', Ori: '{self.ori_key}', UWB: '{self.uwb_key}', Pose: '{self.pose_key}'")

                # Override sequence indices to use ALL data if indices look wrong
                if self.pose_key in raw_data:
                    total_seqs = len(raw_data[self.pose_key])
                    if sequence_indices is None or (sequence_indices and max(sequence_indices) >= total_seqs):
                        print(f"Notice: Using all {total_seqs} sequences from the file.")
                        sequence_indices = list(range(total_seqs))
                else:
                    print(f"FATAL: Could not find pose key '{self.pose_key}' in file keys: {list(raw_data.keys())}")
                    return

            except Exception as e:
                print(f"Error loading single data file: {e}")
                return

        # --- CASE 2: Directory Structure (AMASS Data) ---
        else:
            # Default keys for AMASS
            self.acc_key = ACC_KEY
            self.ori_key = ORI_KEY
            self.uwb_key = UWB_KEY
            self.pose_key = POSE_KEY

            # Create a unique cache file name
            cache_name = f"raw_data_cache_n{len(sequence_indices)}_idx{sequence_indices[0]}.pt"
            
            cache_path = os.path.join(data_path, cache_name)

            if os.path.exists(cache_path):
                print(f"Found cache file. Loading from {cache_path}...")
                try:
                    raw_data = torch.load(cache_path, map_location='cpu')
                    print("Cache loaded successfully.")
                except Exception as e:
                    print(f"Error loading cache file: {e}. Re-building dataset.")
                    raw_data = None
            else:
                print(f"Cache file not found. Building dataset...")
                raw_data = None
                
            if raw_data is None:
                all_pose_files = sorted(glob.glob(os.path.join(data_path, self.pose_key, '*.pt')))
                if not all_pose_files:
                    print(f"FATAL: No files found at {os.path.join(data_path, self.pose_key, '*.pt')}")
                    return

                try:
                    pose_files_to_load = [all_pose_files[i] for i in sequence_indices]
                except IndexError:
                    print(f"FATAL: Error splitting files.")
                    return
                    
                print(f"Loading {len(pose_files_to_load)} sequences...")
                pose_list, vacc_list, vrot_list, vuwb_list = [], [], [], []
                
                for pose_path in tqdm(pose_files_to_load, desc="Loading data"):
                    file_name = os.path.basename(pose_path)
                    vacc_path = os.path.join(data_path, self.acc_key, file_name)
                    vrot_path = os.path.join(data_path, self.ori_key, file_name)
                    vuwb_path = os.path.join(data_path, self.uwb_key, file_name)
                    
                    try:
                        pose_list.append(torch.load(pose_path, map_location='cpu'))
                        vacc_list.append(torch.load(vacc_path, map_location='cpu'))
                        vrot_list.append(torch.load(vrot_path, map_location='cpu'))
                        vuwb_list.append(torch.load(vuwb_path, map_location='cpu'))
                    except Exception as e:
                        print(f"Warning: Skipping sequence {file_name}. Error: {e}")
                        continue
                
                raw_data = {
                    self.pose_key: pose_list,
                    self.acc_key: vacc_list,
                    self.ori_key: vrot_list,
                    self.uwb_key: vuwb_list
                }
                
                if raw_data[self.pose_key]:
                    print(f"\nSaving raw data to cache file: {cache_path} ...")
                    torch.save(raw_data, cache_path)

        # --- PROCESSING ---
        self.limb_lengths_by_seq_idx = {}
        fixed_lengths = torch.tensor([REFERENCE_UPPER_ARM, REFERENCE_FOREARM], dtype=torch.float32)
        
        relevant_indices = sequence_indices if sequence_indices is not None else range(len(raw_data[self.pose_key]))
        
        for i in range(len(raw_data[self.pose_key])):
             self.limb_lengths_by_seq_idx[i] = fixed_lengths

        all_input_sequences = []
        all_target_sequences = []
        all_limb_lengths_continuous = [] 

        self.input_windows = []
        self.target_windows = []
        self.limb_lengths_windows = [] 

        for i in relevant_indices:
            if i >= len(raw_data[self.pose_key]): continue

            limb_lengths = self.limb_lengths_by_seq_idx.get(i, fixed_lengths)

            try:
                # USE DYNAMIC KEYS HERE
                pelvis_acc = raw_data[self.acc_key][i][:, PELVIS_IDX, :]
                wrist_acc = raw_data[self.acc_key][i][:, WRIST_IDX, :]
                pelvis_ori_mat = raw_data[self.ori_key][i][:, PELVIS_IDX, :, :]
                wrist_ori_mat = raw_data[self.ori_key][i][:, WRIST_IDX, :, :]
                uwb_dist = raw_data[self.uwb_key][i][:, PELVIS_IDX, WRIST_IDX].unsqueeze(-1)

                pelvis_ori_flat = pelvis_ori_mat.reshape(-1, 9) 
                wrist_ori_flat = wrist_ori_mat.reshape(-1, 9) 
                
                input_features = torch.cat([ pelvis_acc, pelvis_ori_flat, wrist_acc, wrist_ori_flat, uwb_dist ], dim=1) 

                shldr_pose = raw_data[self.pose_key][i][:, SHLDR_POSE_IDX, :]
                elbow_pose = raw_data[self.pose_key][i][:, ELBOW_POSE_IDX, :]
                target_features = torch.cat([shldr_pose, elbow_pose], dim=1)
                
                input_seq = input_features[::DOWNSAMPLE_RATE]
                target_seq = target_features[::DOWNSAMPLE_RATE]
                num_frames_downsampled = input_seq.shape[0]

                if create_windows: 
                    if num_frames_downsampled >= sequence_length:
                        for start_idx in range(0, num_frames_downsampled - sequence_length + 1, stride):
                            end_idx = start_idx + sequence_length
                            
                            # Grab the window and clone it for mutation
                            input_window_tensor = input_seq[start_idx:end_idx, :].clone()
                            target_window_tensor = target_seq[start_idx:end_idx, :]
                            
                            # --- AUGMENTATION ---
                            if self.augment:
                                input_window_tensor = self._apply_augmentation(input_window_tensor)

                            self.input_windows.append(input_window_tensor)
                            self.target_windows.append(target_window_tensor)
                            self.limb_lengths_windows.append(limb_lengths)
                else:
                    all_input_sequences.append(input_seq)
                    all_target_sequences.append(target_seq)
                    all_limb_lengths_continuous.append(limb_lengths.expand(num_frames_downsampled, 2))

            except (IndexError, KeyError) as e:
                print(f"Warning: Skipping sequence index {i} due to error: {e}")
                continue

        # --- Normalization ---
        if create_windows:
            if not self.input_windows:
                print(f"Warning: No valid windows generated.")
                return 
            temp_inputs_stacked = torch.stack(self.input_windows)
            temp_targets_stacked = torch.stack(self.target_windows)
            self.num_windows = len(self.input_windows)
            feature_dim = self.input_windows[0].shape[1]
        else:
            if not all_input_sequences:
                print(f"Warning: No valid sequences loaded.")
                return 
            self.inputs_continuous = torch.cat(all_input_sequences, dim=0)
            self.targets_continuous = torch.cat(all_target_sequences, dim=0)
            self.limb_lengths_continuous = torch.cat(all_limb_lengths_continuous, dim=0)
            feature_dim = self.inputs_continuous.shape[1]

        # --- Stats Handling ---
        # Priority 1: Use provided stats 
        if stats is not None:
            self.mean = stats['input_mean_std']['mean'].cpu()
            self.std = stats['input_mean_std']['std'].cpu()
            self.target_min = stats['target_min_max']['min'].cpu()
            self.target_max = stats['target_min_max']['max'].cpu()
            self.target_range = self.target_max - self.target_min
            self.target_range[self.target_range == 0] = 1.0
            
        # Priority 2: Calculate from scratch (only if no stats provided AND is_train is True)
        elif is_train:
            if create_windows:
                input_flat = temp_inputs_stacked.view(-1, feature_dim)
                target_flat = temp_targets_stacked.view(-1, OUTPUT_SIZE)
            else:
                input_flat = self.inputs_continuous
                target_flat = self.targets_continuous
            self.mean = input_flat.mean(dim=0)
            self.std = input_flat.std(dim=0)
            self.std[self.std == 0] = 1.0
            self.target_min = target_flat.min(dim=0)[0]
            self.target_max = target_flat.max(dim=0)[0]
            self.target_range = self.target_max - self.target_min
            self.target_range[self.target_range == 0] = 1.0 
        else:
            # Case: Test/Val mode but no stats provided
            raise ValueError("Stats required for val/test if not calculating from scratch.")

        # --- Apply Normalization ---
        if create_windows:
            for i in range(self.num_windows):
                self.input_windows[i] = (self.input_windows[i] - self.mean) / self.std
                self.target_windows[i] = 2 * (self.target_windows[i] - self.target_min) / self.target_range - 1
        else:
            self.inputs = (self.inputs_continuous - self.mean) / self.std
            self.targets = 2 * (self.targets_continuous - self.target_min) / self.target_range - 1
            self.limb_lengths = self.limb_lengths_continuous 

    def _apply_augmentation(self, input_window):
        """
        Applies a random combination of noise and scaling to the input window features.
        """
        
        # Define feature slices based on INPUT_SIZE=25 layout (Pelvis acc, Pelvis ori, Wrist acc, Wrist ori, UWB)
        # 0:3 - Pelvis Acc (3)
        # 3:12 - Pelvis Ori (9)
        # 12:15 - Wrist Acc (3)
        # 15:24 - Wrist Ori (9)
        # 24:25 - UWB (1)

        # 1. Add Gaussian Noise (Time-varying sensor noise)
        accel_indices = torch.tensor([0, 1, 2, 12, 13, 14], dtype=torch.long)
        orient_indices = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype=torch.long)
        dist_index = torch.tensor([24], dtype=torch.long)
        
        # Acceleration Noise
        accel_noise = torch.randn_like(input_window[:, accel_indices]) * self.accel_noise_std
        input_window[:, accel_indices] += accel_noise

        # Orientation Noise
        orient_noise = torch.randn_like(input_window[:, orient_indices]) * self.orient_noise_std
        input_window[:, orient_indices] += orient_noise
        
        # UWB Distance Noise
        dist_noise = torch.randn_like(input_window[:, dist_index]) * self.dist_noise_std
        input_window[:, dist_index] += dist_noise
        
        # 2. Add Bias/Offset (Sensor Zero-offset drift, applied uniformly across the window)
        # Only applied to Accelerometers (indices 0-2 and 12-14)
        accel_offset = torch.randn(1, 6) * self.accel_offset_std # Generate 1 bias per sensor axis
        
        input_window[:, accel_indices] += accel_offset.repeat(input_window.shape[0], 1)

        # 3. Random Scaling (Sensor reading scale factor, applied uniformly)
        scale_factor = (self.scale_range[1] - self.scale_range[0]) * torch.rand(1) + self.scale_range[0]
        input_window *= scale_factor # Scale everything equally
        
        return input_window


    def get_stats(self):
        if not hasattr(self, 'mean'): return None
        return {
            'input_mean_std': {'mean': self.mean, 'std': self.std},
            'target_min_max': {'min': self.target_min, 'max': self.target_max}
        }

    def __len__(self):
        return self.num_windows if self.create_windows else (self.inputs_continuous.shape[0] if hasattr(self, 'inputs_continuous') else 0)

    def __getitem__(self, index):
        if self.create_windows:
            return self.input_windows[index], self.target_windows[index], self.limb_lengths_windows[index]
        else:
          
            return self.inputs[index], self.targets[index], self.limb_lengths[index]
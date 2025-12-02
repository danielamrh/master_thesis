import torch
from torch.utils.data import Dataset
import numpy as np
from config import * 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import os

# rotation_matrix_to_axis_angle (implementation omitted for brevity)
def rotation_matrix_to_axis_angle(R):
    """ Converts rotation matrices to axis-angle vectors. """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    safe_sin = torch.sin(angle).clamp(min=1e-5)
    axis_unnormalized = torch.stack([ R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1] ], dim=-1)
    axis = axis_unnormalized / (2 * safe_sin.unsqueeze(-1))
    return axis * angle.unsqueeze(-1)


class UIPArmPoseDataset(Dataset):
    """
    Dataset class:
    - Generates windowed data segments (windows) PER sequence first.
    - Pools windows from specified sequences.
    - Calculates normalization over pooled windows.
    - Supports custom stride.
    - Uses 25 features, normalizes inputs/targets.
    - Can optionally return continuous data for evaluation.
    """
    def __init__(self, data_path, sequence_length, sequence_indices, stride,
                 is_train=True, stats=None, create_windows=True,
                 augment=False,
                 accel_noise_std=0.03, 
                 orient_noise_std=0.015, 
                 dist_noise_std=0.03,
                 scale_range=(0.8, 1.2), 
                 accel_offset_std=0.05):
        """
        Args:
            data_path (str): Path to the .pt file.
            sequence_length (int): Window size.
            sequence_indices (list): List of sequence indices to load.
            stride (int): Stride for the sliding window when creating windows.
            is_train (bool): Flag for training mode (calculates stats).
            stats (dict): Normalization stats from training set (for val/test).
            create_windows (bool): If True, generates windowed segments.
                                    If False, stores continuous data.
        """
        print(f"Loading data from {data_path} for sequences: {sequence_indices}...")
        self.sequence_length = sequence_length
        self.stride = stride
        self.create_windows = create_windows
        raw_data = torch.load(data_path, map_location='cpu')

        self.augment = augment and is_train
        self.accel_noise_std = accel_noise_std
        self.orient_noise_std = orient_noise_std
        self.dist_noise_std = dist_noise_std
        self.scale_range = scale_range
        self.accel_offset_std = accel_offset_std

        if self.augment:
             print(f"Data augmentation enabled:")

        all_input_sequences = []
        all_target_sequences = []
        self.input_windows = []
        self.target_windows = []

        # --- Step 1: Load and process specified sequences ---
        for i in sequence_indices:
            try:
                # Input: Pelvis & Wrist acc, ori (as flat matrices), UWB dist
                pelvis_acc = raw_data[ACC_KEY][i][:, PELVIS_IDX, :]
                wrist_acc = raw_data[ACC_KEY][i][:, WRIST_IDX, :]
                pelvis_ori_mat = raw_data[ORI_KEY][i][:, PELVIS_IDX, :, :]
                wrist_ori_mat = raw_data[ORI_KEY][i][:, WRIST_IDX, :, :]
                uwb_dist = raw_data[UWB_KEY][i][:, PELVIS_IDX, WRIST_IDX].unsqueeze(-1)

                wrist_acc_wrt_pelvis = torch.matmul(torch.transpose(pelvis_ori_mat, 1, 2), wrist_acc.unsqueeze(-1)).squeeze(-1) # Relative acceleration
                wrist_wrt_pelvis = torch.matmul(torch.transpose(pelvis_ori_mat, 1, 2), wrist_ori_mat) # Relative orientation

                pelvis_ori_flat = pelvis_ori_mat.reshape(-1, 9) # Flatten 3x3 to 9
                wrist_ori_flat = wrist_wrt_pelvis.reshape(-1, 9) # Flatten 3x3 to 9
                
                # Combine all input features
                input_features = torch.cat([ pelvis_acc, pelvis_ori_flat, wrist_acc_wrt_pelvis, wrist_ori_flat, uwb_dist ], dim=1) 

                # Target: Shoulder and Elbow poses in axis-angle
                shldr_pose = raw_data[POSE_KEY][i][:, SHLDR_POSE_IDX, :]
                elbow_pose = raw_data[POSE_KEY][i][:, ELBOW_POSE_IDX, :]
                
                target_features = torch.cat([shldr_pose, elbow_pose], dim=1)
                
                input_seq = input_features[::DOWNSAMPLE_RATE]
                target_seq = target_features[::DOWNSAMPLE_RATE]

                # --- Step 2: Generate windows per sequence OR store continuous ---
                if create_windows: 
                    seq_len_frames = input_seq.shape[0]
                    if seq_len_frames >= sequence_length:
                        # Create windows using the specified stride
                        for start_idx in range(0, seq_len_frames - sequence_length + 1, stride):
                            end_idx = start_idx + sequence_length
                            input_window_tensor = input_seq[start_idx:end_idx, :].clone() 
                            target_window_tensor = target_seq[start_idx:end_idx, :]

                            if self.augment:
                                # 1. Scaling
                                if self.scale_range is not None and self.scale_range[0] < self.scale_range[1]:
                                    # Apply same scale factor to related features for consistency
                                    accel_scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
                                    # Optional: Different scale for distance?
                                    dist_scale = torch.rand(1).item() * (1.05 - 0.95) + 0.95 # e.g., +/- 5%

                                    # Apply scaling
                                    input_window_tensor[:, 0:3] *= accel_scale    # Pelvis Accel
                                    input_window_tensor[:, 12:15] *= accel_scale   # Wrist Accel
                                    input_window_tensor[:, 24:25] *= dist_scale    # UWB Distance

                                # 2. Offset
                                if self.accel_offset_std > 0:
                                    # Generate one offset per window per sensor type
                                    pelvis_offset = torch.randn(3) * self.accel_offset_std
                                    wrist_offset = torch.randn(3) * self.accel_offset_std
                                    # Add offset consistently across the window
                                    input_window_tensor[:, 0:3] += pelvis_offset
                                    input_window_tensor[:, 12:15] += wrist_offset
                                    # No offset applied to orientation or distance typically

                                # 3. Gaussian Noise 
                                # Add noise to Accelerations
                                if self.accel_noise_std > 0:
                                    input_window_tensor[:, 0:3] += torch.randn_like(input_window_tensor[:, 0:3]) * self.accel_noise_std
                                    input_window_tensor[:, 12:15] += torch.randn_like(input_window_tensor[:, 12:15]) * self.accel_noise_std
                                # Add noise to Orientation Matrix Elements
                                if self.orient_noise_std > 0:
                                    input_window_tensor[:, 3:12] += torch.randn_like(input_window_tensor[:, 3:12]) * self.orient_noise_std
                                    input_window_tensor[:, 15:24] += torch.randn_like(input_window_tensor[:, 15:24]) * self.orient_noise_std
                                # Add noise to UWB Distance
                                if self.dist_noise_std > 0:
                                    input_window_tensor[:, 24:25] += torch.randn_like(input_window_tensor[:, 24:25]) * self.dist_noise_std
                                    input_window_tensor[:, 24:25].clamp_(min=0.0) # Ensure distance is non-negative

                            self.input_windows.append(input_window_tensor)
                            self.target_windows.append(target_window_tensor)
                else:
                    # Store the whole sequence for continuous evaluation
                    all_input_sequences.append(input_seq)
                    all_target_sequences.append(target_seq)

            except IndexError:
                print(f"Warning: Sequence index {i} out of bounds. Skipping.")
                continue

        if create_windows:
            if not self.input_windows:
                raise ValueError(f"No valid windows generated for indices {sequence_indices}.")
            print(f"Generated {len(self.input_windows)} total windows from {len(sequence_indices)} sequence(s).")
            # Convert window lists to temp tensors for normalization calculation
            temp_inputs_stacked = torch.stack(self.input_windows)
            temp_targets_stacked = torch.stack(self.target_windows)
            self.num_windows = len(self.input_windows) # Renamed num_snippets
            feature_dim = self.input_windows[0].shape[1]
        else:
            if not all_input_sequences:
                raise ValueError(f"No valid sequences loaded for indices {sequence_indices}.")
            # Concatenate for continuous mode
            self.inputs_continuous = torch.cat(all_input_sequences, dim=0)
            self.targets_continuous = torch.cat(all_target_sequences, dim=0)
            print(f"Stored {self.inputs_continuous.shape[0]} continuous frames.")
            self.num_windows = 0 # Not applicable
            feature_dim = self.inputs_continuous.shape[1]

        print(f"Input feature dimension: {feature_dim}")

        # --- Step 3: Normalization Logic ---
        if is_train:
            # Determine which data to use for stats calculation
            if create_windows:
                input_flat = temp_inputs_stacked.view(-1, feature_dim)
                target_flat = temp_targets_stacked.view(-1, OUTPUT_SIZE)
            else:
                input_flat = self.inputs_continuous
                target_flat = self.targets_continuous

            # Calculate input stats
            self.mean = input_flat.mean(dim=0)
            self.std = input_flat.std(dim=0)
            self.std[self.std == 0] = 1.0

            # --- Calculate target min/max ---
            self.target_min = target_flat.min(dim=0)[0]
            self.target_max = target_flat.max(dim=0)[0]
            self.target_range = self.target_max - self.target_min
            self.target_range[self.target_range == 0] = 1.0 # Avoid divide-by-zero
        else:
            # Load stats
            if stats is None: raise ValueError("Stats required for val/test.");
            
            # Load input stats
            self.mean = stats['input_mean_std']['mean']
            self.std = stats['input_mean_std']['std']
            
            # --- NEW: Load target min/max ---
            if 'target_min_max' not in stats:
                raise ValueError("Stats dict is missing 'target_min_max' for min/max normalization.")
            self.target_min = stats['target_min_max']['min']
            self.target_max = stats['target_min_max']['max']

            # Ensure all stats tensors are on the CPU for normalization
            self.mean = self.mean.to('cpu')
            self.std = self.std.to('cpu')
            self.target_min = self.target_min.to('cpu')
            self.target_max = self.target_max.to('cpu')

            self.target_range = self.target_max - self.target_min
            self.target_range[self.target_range == 0] = 1.0

        # --- Step 4: Apply normalization ---
        if create_windows:
            for i in range(self.num_windows):
                self.input_windows[i] = (self.input_windows[i] - self.mean) / self.std
                # --- Use Min-Max for targets ---
                self.target_windows[i] = 2 * (self.target_windows[i] - self.target_min) / self.target_range - 1
            # Clean up temp tensors
            if 'temp_inputs_stacked' in locals(): del temp_inputs_stacked
            if 'temp_targets_stacked' in locals(): del temp_targets_stacked
        else:
            # Apply to continuous data
            self.inputs = (self.inputs_continuous - self.mean) / self.std
            # --- Use Min-Max for targets ---
            self.targets = 2 * (self.targets_continuous - self.target_min) / self.target_range - 1

    def get_stats(self):
        if not hasattr(self, 'mean'): raise AttributeError("Stats not computed/loaded.")
        stats_dict = {
            'input_mean_std': {'mean': self.mean, 'std': self.std},
            'target_min_max': {'min': self.target_min, 'max': self.target_max}
        }
        # Optionally, include mean/std for targets if they exist, just in case
        # if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
        #     stats_dict['target_mean_std'] = {'mean': self.target_mean, 'std': self.target_std}
        return stats_dict


    def __len__(self):
        # Length is the number of windows generated
        if self.create_windows:
            return self.num_windows
        else:
            # Length is total frames in continuous mode
            return self.inputs_continuous.shape[0] if hasattr(self, 'inputs_continuous') else 0

    def __getitem__(self, index):
        if self.create_windows:
            # Return the pre-generated, normalized window
            if index >= self.num_windows: raise IndexError(f"Index {index} out of bounds.")
            return self.input_windows[index], self.target_windows[index]
        else:
            # Return a single frame (used internally by evaluate.py's loop)
            return self.inputs[index], self.targets[index]
        
    def plot_window(self, window_index, stats=None, plot_dir="."):
        """ Plots a specific data window into 4 subplots, saving to plot_dir. """
        if not self.create_windows: print("Plotting requires window mode."); return
        if window_index < 0 or window_index >= self.num_windows: print(f"Error: index {window_index} out of bounds."); return

        current_stats = stats if stats is not None else self.get_stats()
        if current_stats is None: print("Error: Stats not available."); return
        
        # Check for required stats keys
        if 'input_mean_std' not in current_stats:
             print("Error: 'input_mean_std' missing from stats."); return
        if 'target_min_max' not in current_stats:
             print("Error: 'target_min_max' missing from stats for plotting."); return

        input_window_norm, target_window_norm = self.__getitem__(window_index)
        
        # Denormalize Inputs (Z-Score)
        input_mean = current_stats['input_mean_std']['mean'].cpu().numpy()
        input_std_safe = np.where(current_stats['input_mean_std']['std'].cpu().numpy() == 0, 1e-6, current_stats['input_mean_std']['std'].cpu().numpy())
        input_window = (input_window_norm.numpy() * input_std_safe) + input_mean
        
        # Denormalize Targets (Min-Max)
        target_min = current_stats['target_min_max']['min'].cpu().numpy()
        target_max = current_stats['target_min_max']['max'].cpu().numpy()
        target_range = target_max - target_min
        target_range[target_range == 0] = 1.0 # Match dataset logic
        
        # Apply inverse Min-Max scaling for targets
        target_window = (target_window_norm.numpy() + 1) / 2 * target_range + target_min

        # --- Plotting ---
        time_axis = np.arange(self.sequence_length)
        # Create 4 subplots vertically
        fig, axs = plt.subplots(4, 1, figsize=(15, 18), sharex=True); # Increased height
        fig.suptitle(f'Data Window {window_index} - Denormalized Inputs & Targets', fontsize=16)
        colors_tab10 = plt.cm.tab10
        colors_coolwarm = plt.cm.coolwarm(np.linspace(0, 1, 6))

        # --- Plot 1: Acceleration Inputs ---
        axs[0].set_title('Acceleration Features (Inputs)')
        acc_colors = colors_tab10(np.linspace(0, 0.5, 6)) # Distinct colors for 6 lines
        # Pelvis Accel (Indices 0, 1, 2)
        for i in range(3):
            axs[0].plot(time_axis, input_window[:, i], color=acc_colors[i], label=f'Pelvis Acc {["X","Y","Z"][i]}', lw=1.0)
        # Wrist Accel (Indices 12, 13, 14)
        for i in range(3):
            axs[0].plot(time_axis, input_window[:, i+12], color=acc_colors[i+3], label=f'Wrist Acc {["X","Y","Z"][i]}', lw=1.0)
        axs[0].set_ylabel('Acceleration (m/s^2)'); axs[0].grid(True);
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')

        # --- Plot 2: Orientation Inputs ---
        axs[1].set_title('Orientation Features (Flat Matrix Inputs)')
        ori_colors_pelvis = plt.cm.Greens(np.linspace(0.3, 1.0, 9)) # Colors for pelvis ori
        ori_colors_wrist = plt.cm.Reds(np.linspace(0.3, 1.0, 9)) # Colors for wrist ori
        # Pelvis Ori (Indices 3 to 11)
        for i in range(9):
             axs[1].plot(time_axis, input_window[:, i+3], color=ori_colors_pelvis[i], label=f'Pelvis Ori [{i}]', lw=0.8, alpha=0.9)
        # Wrist Ori (Indices 15 to 23)
        for i in range(9):
             axs[1].plot(time_axis, input_window[:, i+15], color=ori_colors_wrist[i], label=f'Wrist Ori [{i}]', lw=0.8, alpha=0.9)
        axs[1].set_ylabel('Matrix Element Value'); axs[1].grid(True)
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='x-small') # Try 2 columns

        # --- Plot 3: UWB Distance Input ---
        axs[2].set_title('UWB Distance (Input)')
        axs[2].plot(time_axis, input_window[:, 24], color='purple', label='UWB Dist (m)', lw=1.5)
        axs[2].set_ylabel('Distance (m)'); axs[2].grid(True);
        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')

        # --- Plot 4: Target Angles ---
        axs[3].set_title('Target Angles (Axis-Angle)')
        joint_names = ['Shoulder(X)', 'Shoulder(Y)', 'Shoulder(Z)', 'Elbow(X)', 'Elbow(Y)', 'Elbow(Z)']
        for i in range(6):
            axs[3].plot(time_axis, target_window[:, i], color=colors_coolwarm[i], label=f'{joint_names[i]}', lw=1.5)
        axs[3].set_ylabel('Angle (Radians)'); axs[3].grid(True);
        axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
        axs[3].set_xlabel('Time Steps within Window'); # Add x-label only to the last plot

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.96]); # Adjust right margin

        # --- SAVE PLOT TO DIRECTORY ---
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = f"data_window_4plots_{window_index}.png" # New filename
        full_plot_path = os.path.join(plot_dir, plot_filename)
        try:
            plt.savefig(full_plot_path)
            print(f"Saved 4-subplot plot for window {window_index} to {full_plot_path}")
        except Exception as e:
            print(f"Error saving plot {full_plot_path}: {e}")
        finally:
            plt.close(fig) # Ensure figure is closed

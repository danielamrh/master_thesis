import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from config import *
from dataset import UIPArmPoseDataset 
from lstm import ArmPoseLSTM


def denormalize_targets(targets_normalized, stats):
    """
    Converts normalized targets (range [-1, 1]) back to original radians
    using the stats dictionary.
    """
    try:
        target_min = stats['target_min_max']['min']
        target_max = stats['target_min_max']['max']
        target_range = target_max - target_min
        target_range[target_range == 0] = 1.0
    except KeyError:
        raise ValueError("Stats dictionary is missing 'target_min_max' keys.")
    
    # Move stats to the device of the input tensor
    target_min = target_min.to(targets_normalized.device)
    target_range = target_range.to(targets_normalized.device)

    # Inverse operation of: 2 * (X - min) / range - 1
    # (normalized + 1) / 2 * range + min
    targets_denorm = targets_normalized.add(1.0).div(2.0).mul(target_range).add(target_min)
    
    return targets_denorm

def evaluate_continuous(model, test_dataset, stats, sequence_length):
    """
    Performs continuous, frame-by-frame evaluation.
    Requires test_dataset initialized with create_windows=False.
    """
    print("Starting continuous evaluation...")
    model.eval()
         
    if not hasattr(test_dataset, 'inputs') or not hasattr(test_dataset, 'targets'):
         raise AttributeError("Test dataset not initialized in continuous mode (create_windows=False needed).")

    inputs_normalized = test_dataset.inputs 
    targets_normalized = test_dataset.targets
    
    num_frames = len(inputs_normalized) 
    all_predictions_normalized = [] 
    all_targets_normalized = []
    
    if num_frames < sequence_length: 
        print(f"Warning: Test data length ({num_frames}) is less than sequence length ({sequence_length}). No evaluation possible.")
        return np.array([]), np.array([])

    # (Implementation unchanged - omitted for brevity)
    with torch.no_grad():
        for i in range(num_frames - sequence_length + 1):
            
            input_window = inputs_normalized[i : i + sequence_length]
            input_window_batch = input_window.unsqueeze(0).to(DEVICE)
            
            prediction_seq = model(input_window_batch)
            
            # Get the prediction for the *last* frame in the window
            last_frame_prediction_norm = prediction_seq[0, -1, :].cpu()
            
            # Get the corresponding target frame
            target_frame_norm = targets_normalized[i + sequence_length - 1]
            
            all_predictions_normalized.append(last_frame_prediction_norm)
            all_targets_normalized.append(target_frame_norm)
            
    print(f"Evaluation complete. Processed {num_frames - sequence_length + 1} frames.")
    
    if not all_predictions_normalized:
        return np.array([]), np.array([])

    pred_norm_tensor = torch.stack(all_predictions_normalized)
    targets_norm_tensor = torch.stack(all_targets_normalized)

    # --- De-normalize using the new helper function ---
    print("De-normalizing predictions and targets...")
    pred_denorm = denormalize_targets(pred_norm_tensor, stats).numpy()
    targets_denorm = denormalize_targets(targets_norm_tensor, stats).numpy()

    return pred_denorm, targets_denorm

def plot_evaluation(predictions, targets):
    """ 
    Plots evaluation results and calculates MAE (L1 Loss).
    """
    print("Generating evaluation plots...")

    if len(predictions) == 0: 
        print("No predictions to plot."); 
        return
    
    # --- Calculate MAE (L1 Loss) ---
    # This is the Mean Absolute Error per frame, averaged across all 6 joints
    mae_per_frame = np.mean(np.abs(predictions - targets), axis=1)
    
    # This is the final metric: Mean MAE across all frames
    mean_mae = np.mean(mae_per_frame)
    
    print(f"Overall Mean MAE: {mean_mae:.4f} radians ({mean_mae * (180/np.pi):.2f} degrees)")

    # (Plotting implementation unchanged - omitted for brevity)
    num_total_frames = len(predictions); full_time_axis = np.arange(num_total_frames)
    start = max(0, PLOT_WINDOW_START_FRAME); end = min(num_total_frames, PLOT_WINDOW_END_FRAME)
    
    plot_time_axis = full_time_axis[start:end];    
    plot_predictions = predictions[start:end]; 
    plot_targets = targets[start:end]
    print(f"Plotting frames {start} to {end-1}."); 
    
    fig, axs = plt.subplots(OUTPUT_SIZE, 1, figsize=(15, 12), sharex=True)

    joint_names = ['Shoulder(X)', 'Shoulder(Y)', 'Shoulder(Z)', 'Elbow(X)', 'Elbow(Y)', 'Elbow(Z)']
    
    for i in range(OUTPUT_SIZE): 
        # ... (omitted for brevity)
        axs[i].plot(plot_time_axis, plot_targets[:, i], label='GT', color='blue', lw=1.5) 
        axs[i].plot(plot_time_axis, plot_predictions[:, i], label='Pred', color='red', ls='--', lw=1.0) 
        axs[i].set_ylabel(joint_names[i]) 
        axs[i].legend() 
        axs[i].grid(True)
    
    axs[-1].set_xlabel(f'Time (Frames {start}-{end-1})'); 
    
    plot_path = f'{DRIVE_PROJECT_ROOT}/eval_plots'
    os.makedirs(plot_path, exist_ok=True)
    angle_plot_file = os.path.join(plot_path, 'evaluation_continuous_angles_windowed.png')
    
    plt.savefig(angle_plot_file); 
    print(f"Saved windowed angle plot to {angle_plot_file}"); 
    plt.close(fig)
    
    # --- Plot MAE over time ---
    plt.figure(figsize=(15, 5)); 
    plt.plot(full_time_axis, mae_per_frame, label='MAE', lw=0.8); 
    
    plt.axhline(y=mean_mae, color='r', ls='--', label=f'Mean: {mean_mae:.4f}')
    plt.title('Prediction Error (MAE) Over Time'); 
    plt.xlabel('Time (Frames)'); 
    plt.ylabel('MAE (Radians)'); 
    plt.legend(); plt.grid(True); 
    plt.tight_layout(); 
    
    error_plot_file = os.path.join(plot_path, 'evaluation_continuous_error_full.png')
    plt.savefig(error_plot_file); 
    
    print(f"Saved full error plot to {error_plot_file}"); 
    plt.close()


def main():
    print(f"Using device: {DEVICE}")

    checkpoint_dir = MODEL_SAVE_PATH_BASE
    
    epoch_to_evaluate = EVAL_EPOCH 
    model_load_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_to_evaluate}.pth")
    
    print(f"Attempting to load checkpoint: {model_load_path}")
    if not os.path.exists(model_load_path):
        return

    # 1. Load stats from checkpoint
    try:
        checkpoint = torch.load(model_load_path, map_location=DEVICE)
        if 'stats' not in checkpoint: 
            print(f"Error: Checkpoint missing 'stats'.") 
            return
        stats = checkpoint['stats']; print("Loaded normalization stats from checkpoint.")
    except Exception as e: 
        print(f"Error loading stats from checkpoint: {e}"); 
        return

    # 2. Load test data in CONTINUOUS mode
    print("\nLoading test data (continuous mode)...")
    try:
        # (Data loading logic unchanged - omitted for brevity)
        test_data_check = torch.load(TEST_DATASET_PATH, map_location='cpu')
        
        # --- Check data structure (list vs dict) ---
        if isinstance(test_data_check, list):
            num_test_sequences = len(test_data_check)

        elif isinstance(test_data_check, dict):
            # Find the first key that is a list/tuple to determine seq count
            num_test_sequences = 0
            for k, v in test_data_check.items():
                if isinstance(v, (list, tuple)):
                    num_test_sequences = len(v)
                    print(f"Found {num_test_sequences} sequences in test file (based on key '{k}'). Loading all.")
                    break
            if num_test_sequences == 0:
                print("Error: Test data dict has no list/tuple values to count sequences.")
                return
        else:
            return
            
        del test_data_check
        
        test_sequence_indices = list(range(num_test_sequences))

        test_dataset = UIPArmPoseDataset(
            data_path=TEST_DATASET_PATH,
            sequence_length=SEQUENCE_LENGTH,
            sequence_indices=test_sequence_indices, # Load all test sequences
            stride=1, # Irrelevant for continuous mode but needs a value
            is_train=False,
            stats=stats,
            create_windows=False # Load as continuous
        )
        print("Loaded test data in continuous mode.")

    except FileNotFoundError:
        return
    except Exception as e: 
        print(f"Error loading test data: {e}") 
        return

    # 3. Initialize model and load weights
    model = ArmPoseLSTM().to(DEVICE)
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from: {model_load_path}")
        else: 
            return
    except Exception as e:
        print(f"--- Error loading model state dict ---")
        print(f"This often means your model architecture in lstm.py has changed")
        print(f"since this checkpoint was saved.")
        print(f"Model H={model.lstm.hidden_size}, L={model.lstm.num_layers}. Config H={HIDDEN_SIZE}, L={NUM_LAYERS}")
        print(f"Model bidirectional={model.lstm.bidirectional}")
        print(f"Original error: {e}")
        return

    model.eval()

    # 4. Run continuous evaluation
    predictions_denorm, targets_denorm = evaluate_continuous(
        model, test_dataset, stats, SEQUENCE_LENGTH
    )

    # 5. Plot results
    if predictions_denorm.size > 0 and targets_denorm.size > 0:
        plot_evaluation(predictions_denorm, targets_denorm)
    else:
        print("Evaluation produced no results, skipping plotting.")

if __name__ == '__main__':
    main()


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from config import *
from dataset import UIPArmPoseDataset 
from lstm import ArmPoseLSTM

def evaluate_continuous(model, test_dataset, stats, sequence_length):
    """
    Performs continuous, frame-by-frame evaluation.
    Requires test_dataset initialized with create_windows=False.
    (Implementation unchanged - omitted for brevity)
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
    if num_frames < sequence_length: print(f"Warning: Test data too short."); return np.array([]), np.array([])
    with torch.no_grad():
        for i in range(num_frames - sequence_length + 1):
            input_window = inputs_normalized[i : i + sequence_length]
            input_window_batch = input_window.unsqueeze(0).to(DEVICE)
            prediction_seq = model(input_window_batch)
            last_frame_prediction_norm = prediction_seq[0, -1, :].cpu()
            target_frame_norm = targets_normalized[i + sequence_length - 1]
            all_predictions_normalized.append(last_frame_prediction_norm)
            all_targets_normalized.append(target_frame_norm)
    print(f"Generated {len(all_predictions_normalized)} continuous predictions.")
    predictions_norm_np = torch.stack(all_predictions_normalized).numpy(); targets_norm_np = torch.stack(all_targets_normalized).numpy()

    # --- MODIFIED: Load min/max and use inverse formula ---
    target_min = stats['target_min_max']['min'].numpy()
    target_max = stats['target_min_max']['max'].numpy()
    target_range = target_max - target_min
    target_range[target_range == 0] = 1.0 

    predictions_denorm = (predictions_norm_np + 1) / 2 * target_range + target_min
    targets_denorm = (targets_norm_np + 1) / 2 * target_range + target_min

    return predictions_denorm, targets_denorm

def plot_evaluation(predictions, targets):
    """ Plots evaluation results. (Implementation unchanged - omitted) """
    print("Generating evaluation plots...")

    if len(predictions) == 0: 
        print("No predictions to plot."); 
        return
    
    rmse_per_frame = np.sqrt(np.mean((predictions - targets)**2, axis=1)); 
    mean_rmse = np.mean(rmse_per_frame)
    print(f"Overall Mean RMSE: {mean_rmse:.4f} radians ({mean_rmse * (180/np.pi):.2f} degrees)")

    num_total_frames = len(predictions); full_time_axis = np.arange(num_total_frames)
    start = max(0, PLOT_WINDOW_START_FRAME); end = min(num_total_frames, PLOT_WINDOW_END_FRAME)
    
    if start >= end: print(f"Warning: Plot window invalid. Plotting first 500."); start=0; end=min(num_total_frames, 500)
    
    plot_time_axis = full_time_axis[start:end];    
    plot_predictions = predictions[start:end]; 
    plot_targets = targets[start:end]
    print(f"Plotting frames {start} to {end-1}."); 
    
    fig, axs = plt.subplots(OUTPUT_SIZE, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Continuous Eval (Frames {start}-{end-1})', fontsize=16)
    joint_names = ['Shoulder(X)', 'Shoulder(Y)', 'Shoulder(Z)', 'Elbow(X)', 'Elbow(Y)', 'Elbow(Z)']
    
    for i in range(OUTPUT_SIZE): 
        axs[i].plot(plot_time_axis, plot_targets[:, i], label='GT', color='blue', lw=1.5) 
        axs[i].plot(plot_time_axis, plot_predictions[:, i], label='Pred', color='red', ls='--', lw=1.0) 
        axs[i].set_ylabel(joint_names[i]) 
        axs[i].legend() 
        axs[i].grid(True)
    
    axs[-1].set_xlabel(f'Time (Frames {start}-{end-1})'); 
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); 
    plt.savefig(f'{DRIVE_PROJECT_ROOT}/eval_plots/evaluation_continuous_angles_windowed.png'); 
    print("Saved windowed angle plot."); 
    plt.close(fig)
    
    plt.figure(figsize=(15, 5)); 
    plt.plot(full_time_axis, rmse_per_frame, label='RMSE', lw=0.8); 
    plt.axhline(y=mean_rmse, color='r', ls='--', label=f'Mean: {mean_rmse:.4f}')
    plt.title('Prediction Error (RMSE) Over Time'); 
    plt.xlabel('Time (Frames)'); 
    plt.ylabel('RMSE (Radians)'); 
    plt.legend(); plt.grid(True); 
    plt.tight_layout(); 
    plt.savefig(f'{DRIVE_PROJECT_ROOT}/eval_plots/evaluation_continuous_error_full.png'); 
    print("Saved full error plot."); 
    plt.close()


def main():
    print(f"Using device: {DEVICE}")

    checkpoint_dir = MODEL_SAVE_PATH_BASE
    epoch_to_evaluate = NUM_EPOCHS
    model_load_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_to_evaluate}.pth")
    print(f"Attempting to load checkpoint: {model_load_path}")
    if not os.path.exists(model_load_path): 
        print(f"Error: Checkpoint not found.") 
        return

    # 1. Load stats from checkpoint
    try:
        checkpoint = torch.load(model_load_path, map_location=DEVICE)
        if 'stats' not in checkpoint: 
            print(f"Error: Checkpoint missing 'stats'.") 
            return
        stats = checkpoint['stats']; print("Loaded stats from checkpoint.")
    except Exception as e: print(f"Error loading stats: {e}"); return

    # 2. Load test data in CONTINUOUS mode
    print("\nLoading test data (continuous mode)...")
    try:
        test_data_check = torch.load(TEST_DATASET_PATH, map_location='cpu')
        num_test_sequences = len(test_data_check[ACC_KEY]); del test_data_check
        print(f"Found {num_test_sequences} sequences in test file.")
        test_sequence_indices = list(range(num_test_sequences))

        test_dataset = UIPArmPoseDataset(
            data_path=TEST_DATASET_PATH,
            sequence_length=SEQUENCE_LENGTH,
            sequence_indices=test_sequence_indices, # Load all test sequences
            stride=1, # Irrelevant for continuous mode but needs a value
            is_train=False,
            stats=stats,
            create_windows=False
        )
        print("Loaded test data in continuous mode.")

    except FileNotFoundError: 
        print(f"Error: Test file not found.") 
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
            print(f"Error: Checkpoint missing 'model_state_dict'.") 
            return
    except Exception as e:
        print(f"Error loading model state dict. Arch mismatch?")
        print(f"Model H={model.lstm.hidden_size}, L={model.lstm.num_layers}. Config H={HIDDEN_SIZE}, L={NUM_LAYERS}")
        print(f"Original error: {e}")
        return

    model.eval()

    # 4. Run continuous evaluation
    predictions_denorm, targets_denorm = evaluate_continuous(
        model, test_dataset, stats, SEQUENCE_LENGTH
    )

    # 5. Plot results
    plot_evaluation(predictions_denorm, targets_denorm)

if __name__ == '__main__':
    main()


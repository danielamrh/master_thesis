import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import *
from dataset import UIPArmPoseDataset
from lstm import ArmPoseLSTM

def evaluate_continuous(model, test_dataset, stats, sequence_length):
    """
    Performs a continuous, frame-by-frame evaluation on the entire test dataset.
    
    Args:
        model: The trained ArmPoseLSTM model.
        test_dataset: The initialized (but un-windowed) test dataset object.
        stats: The normalization stats dict from the training set.
        sequence_length: The model's input sequence length (e.g., 50).
    """
    
    print("Starting continuous evaluation...")
    model.eval()
    
    # 1. Get the raw (normalized) continuous data from the test dataset object
    # These are the *entire* concatenated sequences from the test file.
    inputs_normalized = test_dataset.inputs
    targets_normalized = test_dataset.targets
    
    num_frames = len(inputs_normalized)
    all_predictions_normalized = []
    all_targets_normalized = []
    
    # 2. Manually perform the sliding window inference
    # We slide one frame at a time to get a prediction for every possible frame
    with torch.no_grad():
        for i in range(num_frames - sequence_length):
            # 1. Prepare the input window
            # Shape: (sequence_length, input_size)
            input_window = inputs_normalized[i : i + sequence_length]
            
            # 2. Add a batch dimension: (1, sequence_length, input_size)
            input_window_batch = input_window.unsqueeze(0).to(DEVICE)
            
            # 3. Get model prediction
            # Output shape: (1, sequence_length, output_size)
            prediction_seq = model(input_window_batch)
            
            # 4. Get the prediction for the *last* frame in the window
            # Shape: (output_size,)
            last_frame_prediction = prediction_seq[0, -1, :].cpu()
            
            # 5. Get the corresponding ground truth for that last frame
            target_frame = targets_normalized[i + sequence_length - 1]
            
            all_predictions_normalized.append(last_frame_prediction)
            all_targets_normalized.append(target_frame)

    print(f"Generated {len(all_predictions_normalized)} continuous predictions.")

    # 3. Convert lists to numpy arrays
    predictions_norm = torch.stack(all_predictions_normalized).numpy()
    targets_norm = torch.stack(all_targets_normalized).numpy()
    
    # 4. Denormalize the data for plotting
    target_mean = stats['target_mean_std']['mean'].numpy()
    target_std = stats['target_mean_std']['std'].numpy()
    
    predictions_denorm = (predictions_norm * target_std) + target_mean
    targets_denorm = (targets_norm * target_std) + target_mean
    
    return predictions_denorm, targets_denorm

def plot_evaluation(predictions, targets):
    """
    Plots the continuous predictions vs. ground truth and the error over time.
    """
    print("Generating evaluation plots...")
    
    # Calculate error
    # We will use Euclidean distance in the 6D angle space
    rmse_per_frame = np.sqrt(np.mean((predictions - targets)**2, axis=1)) # RMSE across 6 features
    mean_rmse = np.mean(rmse_per_frame)
    
    print(f"Overall Mean RMSE (over all frames and all {OUTPUT_SIZE} angles): {mean_rmse:.4f} radians")
    print(f"Overall Mean RMSE in degrees: {mean_rmse * (180/np.pi):.2f} degrees")

    num_frames = len(predictions)
    time_axis = np.arange(num_frames)
    
    # --- Plot 1: Per-Joint Angles ---
    # We have 6 output features: 3 for shoulder, 3 for elbow
    fig, axs = plt.subplots(OUTPUT_SIZE, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Continuous Frame-by-Frame Evaluation (Denormalized Axis-Angle)', fontsize=16)
    
    joint_names = [
        'Shoulder (Axis-X)', 'Shoulder (Axis-Y)', 'Shoulder (Axis-Z)',
        'Elbow (Axis-X)', 'Elbow (Axis-Y)', 'Elbow (Axis-Z)'
    ]
    
    for i in range(OUTPUT_SIZE):
        axs[i].plot(time_axis, targets[:, i], label='Ground Truth', color='blue')
        axs[i].plot(time_axis, predictions[:, i], label='Prediction', color='red', linestyle='--')
        axs[i].set_ylabel(joint_names[i])
        axs[i].legend()
        axs[i].grid(True)
        
    axs[-1].set_xlabel('Time (Frames)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('evaluation_continuous_angles.png')
    print("Saved continuous angle plot to 'evaluation_continuous_angles.png'")

    # --- Plot 2: RMSE Error Over Time ---
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, rmse_per_frame, label='Per-Frame RMSE')
    plt.axhline(y=mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.4f}')
    plt.title('Prediction Error (RMSE) Over Time')
    plt.xlabel('Time (Frames)')
    plt.ylabel('RMSE (Radians)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('evaluation_continuous_error.png')
    print("Saved continuous error plot to 'evaluation_continuous_error.png'")

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load training dataset *only* to get the normalization stats
    print("Loading training data to get normalization stats...")
    train_dataset_for_stats = UIPArmPoseDataset(
        data_path=TRAIN_DATASET_PATH,
        sequence_length=SEQUENCE_LENGTH,
        is_train=True
    )
    stats = train_dataset_for_stats.get_stats()
    
    # 2. Load the test dataset using the training stats
    print("Loading test data...")
    test_dataset = UIPArmPoseDataset(
        data_path=TEST_DATASET_PATH,
        sequence_length=SEQUENCE_LENGTH,
        is_train=False,
        stats=stats
    )

    # 3. Initialize model and load saved weights
    model = ArmPoseLSTM().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        print("Please run train.py to train and save a model first.")
        return
    except Exception as e:
        print(f"Error loading model state. Did you change the model architecture?")
        print(f"Original error: {e}")
        print("You may need to re-train your model if you changed the architecture.")
        return
        
    model.eval()
    print(f"Successfully loaded model from {MODEL_SAVE_PATH}")

    # 4. Run the continuous evaluation
    predictions_denorm, targets_denorm = evaluate_continuous(
        model, 
        test_dataset, 
        stats, 
        SEQUENCE_LENGTH
    )
    
    # 5. Plot the results
    if len(predictions_denorm) > 0:
        plot_evaluation(predictions_denorm, targets_denorm)
    else:
        print("No predictions were generated. Is the test dataset empty or too short?")

if __name__ == '__main__':
    main()
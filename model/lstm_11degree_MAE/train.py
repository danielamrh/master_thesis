import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

from config import *
from dataset import UIPArmPoseDataset # Uses the updated class
from lstm import ArmPoseLSTM

# --- BIOMECHANICAL CONSTRAINTS ---
# Define physiological limits for the 6 output angles (in radians)
# Order: ['Shoulder(X)', 'Shoulder(Y)', 'Shoulder(Z)', 'Elbow(X)', 'Elbow(Y)', 'Elbow(Z)']
LIMITS_MIN = torch.tensor([-torch.inf, -torch.inf, -torch.inf, 0.0, -1.5, -torch.inf], device=DEVICE)
LIMITS_MAX = torch.tensor([ torch.inf,  torch.inf,  torch.inf, 2.7,  1.5,  torch.inf], device=DEVICE)

def main():
    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
    print(f"Using device: {DEVICE}")

    checkpoint_dir = MODEL_SAVE_PATH_BASE 
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {checkpoint_dir}")

    plot_dir = f"{DRIVE_PROJECT_ROOT}/random_windows_plots"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Data inspection plots will be saved in: {plot_dir}")

    # --- 1. Create Datasets (Window Mode) ---
    print(f"Creating training dataset (window mode) using sequences: {TRAIN_SEQUENCE_INDICES}")
    try:
        train_dataset = UIPArmPoseDataset(
            data_path=TRAIN_DATASET_PATH, 
            sequence_length=SEQUENCE_LENGTH, 
            sequence_indices=TRAIN_SEQUENCE_INDICES, 
            stride=STRIDE, 
            is_train=True, 
            create_windows=True,
            augment=True
        )
        stats = train_dataset.get_stats()
    except Exception as e:
        print(f"Error creating training dataset: {e}"); return

    # --- NEW: Get stats for de-normalization ---
    target_min = stats['target_min_max']['min'].to(DEVICE)
    target_max = stats['target_min_max']['max'].to(DEVICE)
    target_range = target_max - target_min
    # Avoid division by zero if a range is 0
    target_range[target_range == 0] = 1.0 
    
    # --- Define constraint weight ---
    # How much to penalize constraint violations
    constraint_weight = 0.5

    # Data Inspection
    print("\n--- Inspecting Random Training Data Windows ---")
    num_windows_to_plot = 3 # How many random windows to check
    if len(train_dataset) > 0:
        num_windows_to_plot = min(num_windows_to_plot, len(train_dataset))
        random_indices = random.sample(range(len(train_dataset)), num_windows_to_plot)
        print(f"Plotting {num_windows_to_plot} random windows: indices {random_indices}")
        for idx in random_indices:
            train_dataset.plot_window(idx, plot_dir=plot_dir)
        print("--- Data Inspection Complete ---\n")
    else:
        print("Training dataset is empty, skipping inspection.")
    print(f"\nCreating validation dataset (window mode) using sequences: {VAL_SEQUENCE_INDICES}")

    try:
        val_dataset = UIPArmPoseDataset(
            data_path=TRAIN_DATASET_PATH,
            sequence_length=SEQUENCE_LENGTH,
            sequence_indices=VAL_SEQUENCE_INDICES,
            stride=STRIDE,
            is_train=False,
            stats=stats,
            create_windows=True
        )
    except ValueError as e: print(f"\nError: {e}"); val_dataset = None
    except Exception as e: print(f"\nUnexpected error: {e}"); val_dataset = None

    # --- 2. Create DataLoaders ---
    # DataLoader works with the list of windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2, 
        pin_memory=True
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        print(f"\nCreated {len(train_loader)} training batches (from pooled windows)")
        print(f"Created {len(val_loader)} validation batches (from pooled windows)")
    else:
        print(f"\nCreated {len(train_loader)} training batches (from pooled windows)")
        print("Warning: No validation data loaded.")

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = ArmPoseLSTM().to(DEVICE) # Initialize model

    # criterion = torch.nn.MSELoss() # Mean Squared Error Loss 
    criterion = torch.nn.L1Loss() # Using L1 Loss

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Adam optimizer with weight decay for regularization 

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    print("Model initialized:"); print(model)

    # --- 4. Training Loop ---
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # Training Phase
        model.train() 
        total_train_loss = 0
        total_train_mae = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # --- MODEL FORWARD PASS ---
            outputs = model(inputs) # Shape: (B, Seq, 6), normalized
            loss_pose = criterion(outputs, targets)
            
            w_constraint = 1.0

            # --- BIOMECHANICAL CONSTRAINT LOSS ---
            # Denormalize all frames to check constraints
            # Ensure target_range and target_min are loaded from stats and sent to DEVICE
            denorm_outputs = (outputs + 1.0) / 2.0 * target_range + target_min
            
            loss_below_min = torch.relu(LIMITS_MIN - denorm_outputs).mean()
            loss_above_max = torch.relu(denorm_outputs - LIMITS_MAX).mean()
            loss_constraint = loss_below_min + loss_above_max

            # --- COMBINE LOSSES & BACKPROP ---
            loss = loss_pose + (w_constraint * loss_constraint)

            optimizer.zero_grad(); 
            loss.backward(); 
            optimizer.step()
            total_train_loss += loss.item() 
            total_train_mae += loss_pose.item()

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_train_mae = total_train_mae / len(train_loader) if len(train_loader) > 0 else 0

        # Validation Phase
        avg_val_loss = 0
        avg_val_mae = 0
        
        if val_loader:
            model.eval() 
            total_val_loss = 0
            total_val_mae = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    
                    loss_pose = criterion(outputs, targets)
                    total_val_mae += loss_pose.item()

                    # --- Calculate constraint loss for validation ---
                    denorm_outputs_val = (outputs + 1.0) / 2.0 * target_range + target_min
                    loss_below_min_val = torch.relu(LIMITS_MIN - denorm_outputs_val).mean()
                    loss_above_max_val = torch.relu(denorm_outputs_val - LIMITS_MAX).mean()
                    loss_constraint_val = loss_below_min_val + loss_above_max_val
                    
                    # --- Final validation loss ---
                    loss_val = loss_pose + (constraint_weight * loss_constraint_val)
                    
                    total_val_loss += loss_val.item()
                    
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            avg_val_mae = total_val_mae / len(val_loader) if len(val_loader) > 0 else 0

        # Step the scheduler based on validation loss
        if val_loader and avg_val_loss > 0:
            scheduler.step(avg_val_loss)

        end_time = time.time(); epoch_duration = end_time - start_time
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss if val_loader and avg_val_loss > 0 else None)
        history['train_rmse'].append(avg_train_mae) # Storing MAE here
        history['val_rmse'].append(avg_val_mae if val_loader and avg_val_loss > 0 else None) # Storing MAE here
        
        val_loss_str = f"{avg_val_loss:.6f}" if val_loader and avg_val_loss > 0 else "N/A"
        val_mae_str = f"{avg_val_mae:.6f}" if val_loader and avg_val_loss > 0 else "N/A"
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss (L1+Const): {avg_train_loss:.6f} | Val Loss (L1+Const): {val_loss_str} | Train MAE: {avg_train_mae:.6f} | Val MAE: {val_mae_str} | Duration: {epoch_duration:.2f}s")

        # Periodic Saving
        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({ 'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'train_loss': avg_train_loss, 'val_loss': avg_val_loss if val_loader and avg_val_loss > 0 else None, 'stats': stats }, checkpoint_path)
            print(f"-> Checkpoint saved to {checkpoint_path}")

    # --- Final Output and Plotting ---
    print("\nTraining complete.")
    # (Final print and plotting logic remains the same - omitted for brevity)
    last_val_loss = history['val_loss'][-1] if history['val_loss'] and history['val_loss'][-1] is not None else None

    if last_val_loss is not None:
         last_mae_rad = last_val_loss
         last_mae_deg = last_mae_rad * (180 / np.pi)
         print(f"Final validation loss (L1+Const) at epoch {NUM_EPOCHS}: {last_val_loss:.6f}")
         print(f"Corresponding Validation MAE: {last_mae_rad:.6f} radians ({last_mae_deg:.2f} degrees)")

    else: print("Training finished.")

    if NUM_EPOCHS > 0:
        print("\nGenerating and saving loss plots...")
        epochs_range = range(1, NUM_EPOCHS + 1); plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(epochs_range, history['train_loss'], label='Training Loss')
        valid_val_loss = [(e, v) for e, v in zip(epochs_range, history['val_loss']) if v is not None]
        if valid_val_loss: epochs_val, losses_val = zip(*valid_val_loss); plt.plot(list(epochs_val), list(losses_val), label='Validation Loss')
        plt.title('Training and Validation Loss (L1 + Constraint)'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2); plt.plot(epochs_range, history['train_rmse'], label='Training MAE') # Changed label
        valid_val_rmse = [(e, v) for e, v in zip(epochs_range, history['val_rmse']) if v is not None]
        if valid_val_rmse: epochs_val_rmse, rmse_val = zip(*valid_val_rmse); plt.plot(list(epochs_val_rmse), list(rmse_val), label='Validation MAE') # Changed label
        plt.title('Training and Validation Error (MAE)'); plt.xlabel('Epochs'); plt.ylabel('Error (Radians)'); plt.legend(); plt.grid(True) # Changed title
        
        plt.tight_layout(); plot_filename = f'{DRIVE_PROJECT_ROOT}/loss_plots/training_loss_plot.png'; plt.savefig(plot_filename); print(f"Loss plot saved to {plot_filename}")



if __name__ == '__main__':
    main()

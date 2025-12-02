import torch
from torch.utils.data import DataLoader, ConcatDataset # Import ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

from config_amass import *
from dataset_amass_mpjpe import UIPArmPoseDataset
from lstm import ArmPoseLSTM
from transformer import ArmPoseTransformer
from utils import denormalize_targets
from kinematics import forward_kinematics 

# --- BIOMECHANICAL CONSTRAINTS (Unchanged) ---
LIMITS_MIN = torch.tensor([-torch.inf, -torch.inf, -torch.inf, 0.0, -1.5, -torch.inf], device=DEVICE)
LIMITS_MAX = torch.tensor([ torch.inf,  torch.inf,  torch.inf, 2.7,  1.5,  torch.inf], device=DEVICE)

def main():
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    print(f"Using device: {DEVICE}")

    checkpoint_dir = MODEL_SAVE_PATH_BASE 
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Model checkpoints will be saved in: {checkpoint_dir}")

    plot_dir = f"{DRIVE_PROJECT_ROOT}/random_windows_plots"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Data inspection plots will be saved in: {plot_dir}")
    
    # --- 1. Load Data (HYBRID LOADING) ---
    print("--- Loading Data ---")
    
    # A. Load AMASS (Synthetic) - The Base
    print("1. Loading AMASS Synthetic Data...")
    amass_train_dataset = UIPArmPoseDataset(
        data_path=TRAIN_DATASET_PATH,
        sequence_length=SEQUENCE_LENGTH,
        sequence_indices=TRAIN_SEQUENCE_INDICES,
        stride=STRIDE,
        is_train=True,
        augment=True,
        create_windows=True
    )
    
    if len(amass_train_dataset) == 0:
        print("\nFATAL: AMASS dataset initialization failed.")
        return
        
    # Extract Stats from AMASS to normalize everything else
    stats = amass_train_dataset.get_stats() 
    
    # B. Load UIP (Real) - The Fine-Tuning Data
    print("\n2. Loading UIP Real Data...")
    # Note: We pass sequence_indices=None to use ALL available real data
    # Note: We pass stats=stats to force it to use AMASS normalization
    # Note: We pass is_train=True so we still get augmentation on the real data
    uip_train_dataset = UIPArmPoseDataset(
        data_path=TRAIN_UIP_DATA_DIR,
        sequence_length=SEQUENCE_LENGTH,
        sequence_indices=None, 
        stride=STRIDE,
        is_train=True,
        augment=True,
        create_windows=True,
        stats=stats 
    )
    
    # C. Combine Datasets
    if len(uip_train_dataset) > 0:
        print(f"\nCombining {len(amass_train_dataset)} AMASS windows + {len(uip_train_dataset)} UIP windows.")
        train_dataset = ConcatDataset([amass_train_dataset, uip_train_dataset])
    else:
        print("\nWarning: UIP dataset empty or not found. Training on AMASS only.")
        train_dataset = amass_train_dataset

    # D. Validation Dataset (Keep as AMASS only for consistent metrics)
    val_dataset = UIPArmPoseDataset(
        data_path=TRAIN_DATASET_PATH,
        sequence_length=SEQUENCE_LENGTH,
        sequence_indices=VAL_SEQUENCE_INDICES,
        stride=STRIDE,
        is_train=False,
        stats=stats,
        create_windows=True
    )
    
    if len(val_dataset) == 0:
        print("\nFATAL: Validation dataset initialization failed.")
        return

    train_loader = DataLoader(train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, # Important to shuffle mixed data!
        pin_memory=True, 
        num_workers=2
    )

    val_loader = DataLoader(val_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=2
    )

    print(f"Total Training Samples: {len(train_dataset)}. Validation Samples: {len(val_dataset)}.")

    # 2. Initialize Model, Optimizer, Scheduler
    #model = ArmPoseLSTM().to(DEVICE)
    model = ArmPoseTransformer().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # 3. Training Loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss, total_train_mae = 0.0, 0.0
        start_time = time.time()

        # --- Data loader now yields limb_lengths_batch ---
        for x_batch, y_batch, limb_lengths_batch in train_loader:
            x_batch, y_batch, limb_lengths_batch = x_batch.to(DEVICE), y_batch.to(DEVICE), limb_lengths_batch.to(DEVICE)
            
            # Extract per-subject lengths
            upper_arm_lens = limb_lengths_batch[:, 0]
            forearm_lens = limb_lengths_batch[:, 1]
            
            optimizer.zero_grad()
            
            pred_angles_norm = model(x_batch) # Shape [batch, 6]
            gt_angles_norm = y_batch[:, -1, :] # Shape [batch, 6]

            # --- Pass limb lengths to FK ---
            pred_positions = forward_kinematics(pred_angles_norm, upper_arm_lens, forearm_lens)
            gt_positions = forward_kinematics(gt_angles_norm, upper_arm_lens, forearm_lens)
            
            loss_pose = F.l1_loss(pred_positions, gt_positions)

            loss_angle = F.l1_loss(pred_angles_norm, gt_angles_norm)

            # Constraint loss 
            pred_angles_denorm = denormalize_targets(pred_angles_norm, stats)
            loss_constraint = F.relu(LIMITS_MIN - pred_angles_denorm) + \
                              F.relu(pred_angles_denorm - LIMITS_MAX)
            loss_constraint = loss_constraint.mean()

            w_constraint = 0.5
            w_angle = 1.0

            # Hybrid loss function MPJPE + joint angle constraints + MAE angle
            loss = loss_pose + (w_constraint * loss_constraint) + (w_angle * loss_angle)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_mae += loss_pose.item() 

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mae = total_train_mae / len(train_loader) 
        history['train_loss'].append(avg_train_loss)
        history['train_mae'].append(avg_train_mae) 

        # Validation Loop
        model.eval()
        total_val_loss, total_val_mae = 0.0, 0.0 
        with torch.no_grad():
            # --- Data loader yields limb_lengths_batch ---
            for x_batch, y_batch, limb_lengths_batch in val_loader:
                x_batch, y_batch, limb_lengths_batch = x_batch.to(DEVICE), y_batch.to(DEVICE), limb_lengths_batch.to(DEVICE)
                
                upper_arm_lens = limb_lengths_batch[:, 0]
                forearm_lens = limb_lengths_batch[:, 1]

                pred_angles_norm = model(x_batch)
                gt_angles_norm = y_batch[:, -1, :]

                # --- Pass limb lengths to FK ---
                pred_positions = forward_kinematics(pred_angles_norm, upper_arm_lens, forearm_lens)
                gt_positions = forward_kinematics(gt_angles_norm, upper_arm_lens, forearm_lens)
                
                loss_pose = F.l1_loss(pred_positions, gt_positions)
                loss_angle = F.l1_loss(pred_angles_norm, gt_angles_norm)
                
                # Constraint loss 
                pred_angles_denorm = denormalize_targets(pred_angles_norm, stats)
                loss_constraint = F.relu(LIMITS_MIN - pred_angles_denorm) + \
                                  F.relu(pred_angles_denorm - LIMITS_MAX)
                loss_constraint = loss_constraint.mean()

                w_constraint = 0.5
                w_angle = 1.0
                loss = loss_pose + (w_constraint * loss_constraint) + (w_angle * loss_angle)
                
                total_val_loss += loss.item()
                total_val_mae += loss_pose.item() 

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mae = total_val_mae / len(val_loader)
        scheduler.step(avg_val_loss) 
        
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae) 
        
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{NUM_EPOCHS} [{epoch_time:.1f}s] - "
              f"Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - "
              f"Train MAE (pos): {avg_train_mae:.6f} - Val MAE (pos): {avg_val_mae:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(checkpoint_dir, "best_model_mpjpe_transformer_amass_uip_120frames.pt") 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'stats': stats 
            }, save_path)
            print(f"  -> New best model saved to {save_path}")

    print("Training finished.")

    # --- Plotting ---
    if NUM_EPOCHS > 0:
        print("\nGenerating and saving loss plots...")
        epochs_range = range(1, NUM_EPOCHS + 1)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], label='Training Loss (Hybrid)')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss (Hybrid)')
        plt.title('Training and Validation Loss (Positional + Constraint)')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_mae'], label='Training MAE (Positional)')
        plt.plot(epochs_range, history['val_mae'], label='Validation MAE (Positional)')
        plt.title('Training and Validation MAE (Positional Only)')
        plt.xlabel('Epochs'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plot_save_path = os.path.join(checkpoint_dir, "loss_plot_mpjpe.png")
        plt.savefig(plot_save_path)
        print(f"Loss plots saved to {plot_save_path}")

if __name__ == "__main__":
    main()
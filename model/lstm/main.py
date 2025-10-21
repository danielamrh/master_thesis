import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import matplotlib.pyplot as plt

from config import *
from dataset import UIPArmPoseDataset
from lstm import ArmPoseLSTM

def main():
    # --- History tracking for plots ---
    history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
    
    print(f"Using device: {DEVICE}")

    # --- 1. Create and Split Dataset ---
    full_train_dataset = UIPArmPoseDataset(
        data_path=TRAIN_DATASET_PATH,
        sequence_length=SEQUENCE_LENGTH,
        is_train=True
    )
    
    total_size = len(full_train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size

    print(f"Splitting dataset of size {total_size} into {train_size} training and {val_size} validation samples.")

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- 2. Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created {len(train_loader)} training batches and {len(val_loader)} validation batches.")

    # --- 3. Initialize Model, Loss, and Optimizer ---
    model = ArmPoseLSTM().to(DEVICE)
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Model initialized:")
    print(model)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        with torch.no_grad(): 
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        # --- Calculate RMSE and store history ---
        avg_train_rmse = np.sqrt(avg_train_loss)
        avg_val_rmse = np.sqrt(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(avg_train_rmse)
        history['val_rmse'].append(avg_val_rmse)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss (MSE): {avg_train_loss:.6f} | "
              f"Val Loss (MSE): {avg_val_loss:.6f} | "
              f"Train RMSE: {avg_train_rmse:.6f} | "
              f"Val RMSE: {avg_val_rmse:.6f} | "
              f"Duration: {epoch_duration:.2f}s")
        
        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Corresponding RMSE: {np.sqrt(best_val_loss):.6f} radians ({np.sqrt(best_val_loss)*(180/3.14159):.2f} degrees)")
    
    # --- Plotting Loss Curves ---
    print("Generating and saving loss plots...")
    epochs_range = range(1, NUM_EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot MSE Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_rmse'], label='Training RMSE')
    plt.plot(epochs_range, history['val_rmse'], label='Validation RMSE')
    plt.title('Training and Validation Error (RMSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Error (Radians)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_filename = 'training_loss_plot.png'
    plt.savefig(plot_filename)
    print(f"Loss plot saved to {plot_filename}")

if __name__ == '__main__':
    main()
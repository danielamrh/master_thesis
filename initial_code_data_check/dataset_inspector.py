import torch

# Path to your PyTorch dataset
pkl_path = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"

# Load dataset on CPU
data = torch.load(pkl_path, map_location=torch.device('cpu'))

print("==== Dataset Overview ====\n")
print("Keys in dataset:", data.keys(), "\n")

# Iterate through all fields
for key in data.keys():
    value = data[key]
    
    if isinstance(value, list):
        print(f"{key}: List, length={len(value)}")
        for i, seq in enumerate(value):
            print(f"  Sequence {i} shape: {seq.shape}, dtype={seq.dtype}")
            # Flatten first sequence to show first 10 values
            if seq.numel() > 0:
                flat = seq.flatten()
                print(f"    Sample values (first 10): {flat[:10].tolist()}")
                print(f"    Min: {flat.min().item():.4f}, Max: {flat.max().item():.4f}")
        print()
    elif isinstance(value, torch.Tensor):
        print(f"{key}: Tensor, shape={value.shape}, dtype={value.dtype}")
        flat = value.flatten()
        print(f"  Sample values (first 10): {flat[:10].tolist()}")
        print(f"  Min: {flat.min().item():.4f}, Max: {flat.max().item():.4f}\n")
    else:
        print(f"{key}: type={type(value)}, value sample={value if isinstance(value, str) else '...'}\n")

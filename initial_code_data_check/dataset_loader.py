import os
import torch
from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_DB_Dataset/train.pt"
SMPL_MODEL_DIR = "/home/danielamrhein/master_thesis/data/smplx_models"

# Environment variable for SMPLX
os.environ["SMPLX_MODEL_DIR"] = SMPL_MODEL_DIR

# -----------------------------
# Load dataset
# -----------------------------
print(f"Loading dataset from {DATASET_PATH}...")
data = torch.load(DATASET_PATH, map_location="cpu")

print("\n=== Dataset Overview ===")
for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: tensor, shape={value.shape}, dtype={value.dtype}")
        print(f"  sample values: {value.flatten()[:10]}\n")
    elif isinstance(value, list):
        print(f"{key}: list, length={len(value)}")
        if len(value) > 0:
            if isinstance(value[0], torch.Tensor):
                print(f"  first element shape: {value[0].shape}")
            else:
                print(f"  first element type: {type(value[0])}")
        print()
    else:
        print(f"{key}: {type(value)}\n")

if "pose" not in data:
    raise ValueError("Dataset does not contain 'pose' key!")

poses_list = data["pose"]
print(f"Number of sequences: {len(poses_list)}")
print("========================\n")

# -----------------------------
# Viewer setup
# -----------------------------
C.window_type = "pyqt5"
v = Viewer()

# -----------------------------
# Create SMPL layer
# -----------------------------
smpl_layer = SMPLLayer(
    model_type="smplx",  # use smplx for your dataset
    gender="male"
)

NUM_BODY_JOINTS = smpl_layer.bm.NUM_BODY_JOINTS  # usually 21 for SMPLX body

# -----------------------------
# Visualization parameters
# -----------------------------
DOWNSAMPLE_RATE = 20  # render every 20th frame to save memory

# -----------------------------
# Add sequences
# -----------------------------
for seq_idx, seq_array in enumerate(poses_list):
    print(f"Visualizing sequence {seq_idx} with original shape {seq_array.shape}")

    # Downsample frames
    seq_array_ds = seq_array[::DOWNSAMPLE_RATE, :NUM_BODY_JOINTS, :]
    seq_tensor = torch.tensor(seq_array_ds, dtype=torch.float32)

    num_frames, num_joints, dims = seq_tensor.shape
    seq_tensor_flat = seq_tensor.view(num_frames, num_joints * dims)

    # Create SMPLSequence
    seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=seq_tensor_flat,
        name=f"seq_{seq_idx}_downsampled"
    )

    v.scene.add(seq)

# -----------------------------
# Run viewer
# -----------------------------
v.run()

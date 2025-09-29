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

if "pose" not in data:
    raise ValueError("Dataset does not contain 'pose' key!")

poses_list = data["pose"]
print(f"Number of sequences: {len(poses_list)}")

# -----------------------------
# Pick ONE sequence
# -----------------------------
SEQ_INDEX = 0   # change this to select another sequence
seq_array = poses_list[SEQ_INDEX]
print(f"Selected sequence {SEQ_INDEX} with shape {seq_array.shape}")

# -----------------------------
# Viewer setup
# -----------------------------
C.window_type = "pyqt5"
v = Viewer()

# -----------------------------
# Create SMPL layer
# -----------------------------
smpl_layer = SMPLLayer(
    model_type="smplx",  # SMPL-X model
    gender="male"
)

NUM_BODY_JOINTS = smpl_layer.bm.NUM_BODY_JOINTS  # usually 21 for SMPLX

# -----------------------------
# Visualization settings
# -----------------------------
USE_DOWNSAMPLE = True    # True = show whole sequence with fewer frames
DOWNSAMPLE_RATE = 4     # show every 10th frame
USE_CROP = False         # True = crop to first N frames
MAX_FRAMES = 2000        # how many frames to keep if cropping

# -----------------------------
# Process sequence
# -----------------------------
if USE_DOWNSAMPLE:
    seq_array_proc = seq_array[::DOWNSAMPLE_RATE, :NUM_BODY_JOINTS, :]
    print(f"→ using downsampled sequence with {seq_array_proc.shape[0]} frames")
elif USE_CROP:
    seq_array_proc = seq_array[:MAX_FRAMES, :NUM_BODY_JOINTS, :]
    print(f"→ using cropped sequence with {seq_array_proc.shape[0]} frames")
else:
    seq_array_proc = seq_array[:, :NUM_BODY_JOINTS, :]
    print(f"→ using full sequence with {seq_array_proc.shape[0]} frames (may be heavy!)")

seq_tensor = torch.tensor(seq_array_proc, dtype=torch.float32)
num_frames, num_joints, dims = seq_tensor.shape
seq_tensor_flat = seq_tensor.view(num_frames, num_joints * dims)

# -----------------------------
# Create SMPLSequence
# -----------------------------
seq = SMPLSequence(
    smpl_layer=smpl_layer,
    poses_body=seq_tensor_flat,
    name=f"seq_{SEQ_INDEX}"
)
v.scene.add(seq)

# -----------------------------
# Run viewer
# -----------------------------
v.run()

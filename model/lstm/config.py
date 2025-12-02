import torch

# -- PATHS --
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/lstm"

# AMASS Dataset
TRAIN_DATASET_PATH_AMASS = f"/content/drive/MyDrive/preprocess/data/AMASS_processed_train"
TEST_DATASET_PATH_AMASS = f"/content/drive/MyDrive/preprocess/data/AMASS_processed_test/test_split"

# UIP-Dataset
TRAIN_DATASET_PATH_UIP = f"{DRIVE_PROJECT_ROOT}/data/train.pt"
TEST_DATASET_PATH_UIP = f"{DRIVE_PROJECT_ROOT}/data/test.pt"
MODEL_SAVE_PATH_BASE = f"{DRIVE_PROJECT_ROOT}/model_checkpoints" 

# -- COMPUTATION --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- DATASET PARAMETERS --
TARGET_ARM = 'left'
DOWNSAMPLE_RATE = 5
SEQUENCE_LENGTH = 40 # Window size 
STRIDE = 20          # Stride for sliding window

# --- Define the fixed split (assuming 22 sequences total in train.pt) ---
TRAIN_SEQUENCE_INDICES = list(range(20))
VAL_SEQUENCE_INDICES = [20, 21]

# --- MODEL HYPERPARAMETERS --
INPUT_SIZE = 25    # Using 25 features (acc, ori, uwb)
OUTPUT_SIZE = 6    # Predicting 6 joint angles (shoulder + elbow)
DROPOUT = 0.1      # Dropout rate 

# --- LSTM HYPERPARAMETERS
HIDDEN_SIZE = 128  # LSTM hidden state size
NUM_LAYERS = 3     # Number of LSTM layers

# --- TRANSFORMER PARAMETERS ---
EMBED_DIM = 128               # Dimension of the model (must be divisible by NUM_HEADS)
NUM_HEADS = 8                 # Number of self-attention heads
NUM_TRANSFORMER_LAYERS = 6    # Number of Transformer encoder layers

# -- TRAINING PARAMETERS --
BATCH_SIZE = 128      # Batch size for training and validation 
# LEARNING_RATE = 1e-04 # Learning rate for optimizer LSTM
LEARNING_RATE = 1e-05 # Learning rate for optimizer Transformer
NUM_EPOCHS = 50       # Number of training epochs 

# -- EVALUATION PLOTTING PARAMETERS --
PLOT_WINDOW_START_FRAME = 0
PLOT_WINDOW_END_FRAME = 1000

# -- DATA KEYS AND INDICES --
ACC_KEY = 'acc'
ORI_KEY = 'ori'
UWB_KEY = 'vuwb'     
POSE_KEY = 'pose'    

REFERENCE_UPPER_ARM = 0.30
REFERENCE_FOREARM = 0.27

# --- Define indices for GT_JOINTS (SMPL layout) ---
if TARGET_ARM == 'left':
    PELVIS_IDX, WRIST_IDX = 5, 0        # IMU/UWB Sensor Indices
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16, 18 # SMPL Pose Vector Indices
else: # right arm
    PELVIS_IDX, WRIST_IDX = 5, 1
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 17, 19

# --- Evaluation Specific ---
EVAL_EPOCH = NUM_EPOCHS


import torch

# -- PATHS --
# TRAIN_DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/train.pt"
# TEST_DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
# MODEL_SAVE_PATH_BASE = "model_checkpoints" 

# -- PATHS --
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/lstm"

TRAIN_DATASET_PATH = f"{DRIVE_PROJECT_ROOT}/data/train.pt"
TEST_DATASET_PATH = f"{DRIVE_PROJECT_ROOT}/data/test.pt"
MODEL_SAVE_PATH_BASE = f"{DRIVE_PROJECT_ROOT}/model_checkpoints" 

# -- COMPUTATION --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- DATASET PARAMETERS --
TARGET_ARM = 'left'
DOWNSAMPLE_RATE = 5
SEQUENCE_LENGTH = 40 # Window size
STRIDE = 10          # Stride for sliding window

# --- Define the fixed split (assuming 22 sequences total in train.pt) ---
TRAIN_SEQUENCE_INDICES = list(range(20))
VAL_SEQUENCE_INDICES = [20, 21]

# -- MODEL HYPERPARAMETERS --
INPUT_SIZE = 25    # Using 25 features (acc, ori, uwb)
OUTPUT_SIZE = 6    # Predicting 6 joint angles (shoulder + elbow)
HIDDEN_SIZE = 64   # LSTM hidden state size
NUM_LAYERS = 2     # Number of LSTM layers
DROPOUT = 0.6      # Dropout rate 

# -- TRAINING PARAMETERS --
BATCH_SIZE = 128      # Batch size for training and validation 
LEARNING_RATE = 1e-04 # Learning rate for optimizer 
NUM_EPOCHS = 50       # Number of training epochs 

# -- EVALUATION PLOTTING PARAMETERS --
PLOT_WINDOW_START_FRAME = 0
PLOT_WINDOW_END_FRAME = 1000

# -- DATA KEYS AND INDICES --
ACC_KEY = 'acc'
ORI_KEY = 'ori'
UWB_KEY = 'vuwb'
POSE_KEY = 'pose'

if TARGET_ARM == 'left':
    PELVIS_IDX, WRIST_IDX = 5, 0
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 15, 17
else: # right arm
    PELVIS_IDX, WRIST_IDX = 5, 1
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16, 18

# --- Evaluation Specific ---
EVAL_EPOCH = NUM_EPOCHS # Evaluate the checkpoint from the last epoch by default


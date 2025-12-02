import torch
import glob
import os
import random 

# -- PATHS --
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/lstm"

# --- Define separate paths for Train and Test data ---
FULL_TRAIN_DATA_DIR = f"/content/drive/MyDrive/preprocess/data/AMASS_processed_train" 
TEST_DATA_DIR = f"/content/drive/MyDrive/preprocess/data/AMASS_processed_test/test_split"
TRAIN_UIP_DATA_DIR = f"{DRIVE_PROJECT_ROOT}/data/train.pt"
TEST_UIP_DATA_DIR = f"{DRIVE_PROJECT_ROOT}/data/test.pt"

TRAIN_DATASET_PATH = FULL_TRAIN_DATA_DIR
VAL_DATASET_PATH = FULL_TRAIN_DATA_DIR
TEST_DATASET_PATH = TEST_DATA_DIR

MODEL_SAVE_PATH_BASE = f"{DRIVE_PROJECT_ROOT}/model_checkpoints" 

# -- COMPUTATION --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- DATASET PARAMETERS --
TARGET_ARM = 'left'
DOWNSAMPLE_RATE = 5
SEQUENCE_LENGTH = 120 # Window size 
STRIDE = 60           # Stride for sliding window

# --- Set a random seed for a deterministic train/val split ---
random.seed(42) 

# --- AUGMENTATION PARAMETERS (External Noise) ---
# Tuned to increase robustness against real sensor noise and drift
ACCEL_NOISE_STD = 0.05
ORIENT_NOISE_STD = 0.03
DIST_NOISE_STD = 0.05
SCALE_RANGE = (0.9, 1.1)
ACCEL_OFFSET_STD = 0.05


# --- Split the FULL AMASS dataset for Training/Validation ---
try:
    all_train_files = glob.glob(os.path.join(TRAIN_DATASET_PATH, 'pose', '*.pt'))
    TOTAL_SEQUENCES = len(all_train_files)
    
    if TOTAL_SEQUENCES == 0:
        print(f"WARNING: No training files found in {TRAIN_DATASET_PATH}.")
        TRAIN_SEQUENCE_INDICES = []
        VAL_SEQUENCE_INDICES = []
    else:
        print(f"Found {TOTAL_SEQUENCES} total sequences in {TRAIN_DATASET_PATH} for training.")
        
        all_indices = list(range(TOTAL_SEQUENCES))
        #random.shuffle(all_indices) # This shuffle is now deterministic
        
        val_count = min(max(50, int(TOTAL_SEQUENCES * 0.1)), 500) 
        train_count = TOTAL_SEQUENCES - val_count
        
        VAL_SEQUENCE_INDICES = all_indices[:val_count]
        TRAIN_SEQUENCE_INDICES = all_indices[val_count:]

        print(f"Splitting into {len(TRAIN_SEQUENCE_INDICES)} train and {len(VAL_SEQUENCE_INDICES)} val sequences (deterministic).")
    
except FileNotFoundError:
    print(f"Warning: Could not find data at {TRAIN_DATASET_PATH}. Using default empty split.")
    TRAIN_SEQUENCE_INDICES = []
    VAL_SEQUENCE_INDICES = []


# --- MODEL HYPERPARAMETERS --
INPUT_SIZE = 25
OUTPUT_SIZE = 6
DROPOUT = 0.1 

# --- LSTM HYPERPARAMETERS
HIDDEN_SIZE = 256
NUM_LAYERS = 3

# --- TRANSFORMER PARAMETERS ---
EMBED_DIM = 256
NUM_HEADS = 8
NUM_TRANSFORMER_LAYERS = 6

# -- TRAINING PARAMETERS --
BATCH_SIZE = 256
LEARNING_RATE = 1e-05 # change for transformer to 1e-05 / 1e-04 for LSTM
NUM_EPOCHS = 250

# -- EVALUATION PLOTTING PARAMETERS --
PLOT_WINDOW_START_FRAME = 0
PLOT_WINDOW_END_FRAME = 1000

# -- DATA KEYS AND INDICES --
ACC_KEY = 'vacc'
ORI_KEY = 'vrot'
UWB_KEY = 'vuwb'     
POSE_KEY = 'pose'    

REFERENCE_UPPER_ARM = 0.30
REFERENCE_FOREARM = 0.27

# --- Define indices for GT_JOINTS (SMPL layout) ---
if TARGET_ARM == 'left':
    PELVIS_IDX, WRIST_IDX = 5, 0
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16, 18
else: # right arm
    PELVIS_IDX, WRIST_IDX = 5, 1
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 17, 19

# --- Evaluation Specific ---
EVAL_EPOCH = NUM_EPOCHS
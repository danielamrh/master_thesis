import torch

# -- PATHS --
TRAIN_DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/train.pt"
TEST_DATASET_PATH = "/home/danielamrhein/master_thesis/UIP_dataset/UIP_DB_Dataset/test.pt"
MODEL_SAVE_PATH = "arm_pose_model.pth"

# -- COMPUTATION --
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -- DATASET PARAMETERS --
TARGET_ARM = 'left'  # 'left' or 'right'
DOWNSAMPLE_RATE = 5  # Use every 5th frame to reduce sequence length

# -- MODEL HYPERPARAMETERS --
INPUT_SIZE = 25    # 6 (acc) + 18 (ori as flattened 3x3) + 1 (uwb)
# INPUT_SIZE = 13    # 6 (acc) + 6 (ori as flattened 3x3) + 1 (uwb)
OUTPUT_SIZE = 6    # 3 (shoulder) + 3 (elbow) joint angles to predict
HIDDEN_SIZE = 64   # Number of neurons in the LSTM hidden layer
NUM_LAYERS = 2     # Number of stacked LSTM layers
DROPOUT = 0.1      # Dropout rate for regularization

# -- TRAINING PARAMETERS --
SEQUENCE_LENGTH = 100   # How many time steps to look back (100 frames * 5 downsample = 500 original frames = 5 seconds)
BATCH_SIZE = 64        # Number of sequences per training batch
LEARNING_RATE = 0.001  # Adam optimizer learning rate
NUM_EPOCHS = 50        # Total number of training epochs

# -- DATA KEYS AND INDICES --
# Do not change these unless the dataset format changes
ACC_KEY = 'acc'
ORI_KEY = 'ori'
UWB_KEY = 'vuwb'
POSE_KEY = 'pose'

# Based on UIP dataset structure
if TARGET_ARM == 'left':
    PELVIS_IDX, WRIST_IDX = 5, 0
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 15, 17
else: # right arm
    PELVIS_IDX, WRIST_IDX = 5, 1
    SHLDR_POSE_IDX, ELBOW_POSE_IDX = 16, 18

# -- EVALUATION PLOTTING PARAMETERS --
PLOT_WINDOW_START_FRAME = 1000  # Start plotting from this frame
PLOT_WINDOW_END_FRAME = 1500    # Stop plotting at this frame (adjust as needed)
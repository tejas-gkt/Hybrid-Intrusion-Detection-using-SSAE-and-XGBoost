# config.py

# Hyperparameters from Table 4 of the paper
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001    # Lambda (Weight attenuation)
SPARSITY_PARAM = 0.05   # Rho (Sparsity parameter)
SPARSITY_WEIGHT = 0.1   # Beta (Sparsity weight)
EPOCHS = 10
BATCH_SIZE = 1028

# Architecture Settings
HIDDEN_LAYERS = [64, 32, 16, 8] 

# Data Settings (MODIFIED TO USE SEPARATE FILES)
# !!! IMPORTANT: UPDATE THESE PATHS !!!
TRAIN_FILE = '/home/hdn/Documents/NITK/sem1/IPC/proj1/train_compressed.csv'
TEST_FILE = '/home/hdn/Documents/NITK/sem1/IPC/proj1/CICIOT23/test/test.csv'
EVAL_FILE = '/home/hdn/Documents/NITK/sem1/IPC/proj1/CICIOT23/validation/validation.csv' # Added for completeness, but TEST_FILE is used for evaluation in main.py

TARGET_COLUMN = 'label'
RANDOM_SEED = 42
# Removed TEST_SIZE as data is pre-split
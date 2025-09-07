# Data Parameters
SEQ_LEN     = 24 * 6 * 3
BATCH_SIZE  = 16
FREQ        = 'd'
SCALER_PATH = 'scalers/wt_failure_scaler_vfinal.joblib'
ROOT_PATH   = 'data/sample'

# Model Parameters
DISTIL      = True
E_LAYERS    = 2
D_MODEL     = 512
N_HEADS     = 32
ATTN        = 'full'
EMBED       = 'timeF'
POOLING     = 'mean'
N_CLASSES   = 2
ENC_IN      = 30
CODE_VOCAB_SIZE = 311

# Paths
CKPT_DIR       = 'checkpoints'

# Training Parameters
PATIENCE       = 10
NUM_EPOCHS     = 1
LEARNING_RATE  = 1e-5
WEIGHT_DECAY   = 1e-5

# Inference Parameters
LOOKAHEAD_DAYS = 10
CKPT_PATH      = "checkpoints/model_epoch_1.pth"
CSV_PATH       = "predicted_failures.csv"



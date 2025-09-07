import os, torch
from torch.utils.data import DataLoader, random_split
from models.model import WTFailureClassifier
from datasets.data_loader import WindTurbineFailureDatasetMultiCode
from utils.train_inference_utils import balance_dataset, train
from utils.tools import compute_scaling_parameters
import params 

# #Uncomment to compute and save new scaling parameters

# scaler = compute_scaling_parameters(
#     data_path="datasets/multicode_all", 
#     save_path="scalers/wt_failure_scaler_vfinal.joblib"
# )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
workers = min(8, os.cpu_count() or 1)

os.makedirs(params.CKPT_DIR, exist_ok=True)

train_val_ds = WindTurbineFailureDatasetMultiCode(
    root_path=params.ROOT_PATH,
    window_size=params.SEQ_LEN,
    freq=params.FREQ,
    stride=48,
    scaler_path=params.SCALER_PATH
)

test_ds = WindTurbineFailureDatasetMultiCode(
    root_path=params.ROOT_PATH,
    window_size=params.SEQ_LEN,
    freq=params.FREQ,
    stride=24,
    scaler_path=params.SCALER_PATH
)

print(f'Train+Val samples: {len(train_val_ds):,}   |   Test samples: {len(test_ds):,}')

# Balanced dataset
balanced_dataset = balance_dataset(
    dataset=train_val_ds,
    undersample_ratio=0.9,
    label_index=3,
    random_seed=None
)

train_len = int(0.75 * len(balanced_dataset))
val_len = len(balanced_dataset) - train_len
train_ds, val_ds = random_split(balanced_dataset, [train_len, val_len])

train_loader = DataLoader(
    train_ds,
    batch_size=params.BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=workers,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=params.BATCH_SIZE,
    shuffle=False,
    num_workers=workers,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=params.BATCH_SIZE,
    shuffle=False,
    num_workers=workers,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
)

print(f'Train batches: {len(train_loader)},  Val batches: {len(val_loader)}')
print(f'Test samples: {len(test_ds):,}   |   Test batches: {len(test_loader):,}')

# Count labels
num_pos = sum(train_val_ds[i][3].item() for i in range(len(train_val_ds)))
num_total = len(balanced_dataset)

# Model
model = WTFailureClassifier(
    enc_in=params.ENC_IN,
    n_classes=params.N_CLASSES,
    freq=params.FREQ,
    e_layers=params.E_LAYERS,
    distil=params.DISTIL,
    attn=params.ATTN,
    embed=params.EMBED,
    n_heads=params.N_HEADS,
    d_model=params.D_MODEL,
    code_vocab_size=params.CODE_VOCAB_SIZE,
    pooling=params.POOLING,
).to(device)

# Train
train(
    model,
    train_loader, val_loader, test_loader,
    device=device,
    num_epochs=params.NUM_EPOCHS,
    patience=params.PATIENCE,
    learning_rate=params.LEARNING_RATE,
    weight_decay=params.WEIGHT_DECAY,
    ckpt_dir=params.CKPT_DIR
)

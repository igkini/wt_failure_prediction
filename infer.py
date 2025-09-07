import os, torch
from torch.utils.data import DataLoader
from models.model import WTFailureClassifier
from datasets.data_loader import WindTurbinePredictionWindowDataset
from utils.train_inference_utils import run_inference
import params

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
workers = min(8, os.cpu_count() or 1)

prediction_ds=WindTurbinePredictionWindowDataset(
    root_path=params.ROOT_PATH,
    window_size=params.SEQ_LEN,
    freq=params.FREQ,
    stride=12,
    scaler_path=params.SCALER_PATH,
)

prediction_loader = DataLoader(
    prediction_ds,
    batch_size=params.BATCH_SIZE,
    shuffle=False,
    num_workers=workers,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True,
)

print(f'Prediction samples: {len(prediction_ds):,}   |   Prediction batches: {len(prediction_loader):,}')

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

predicted = run_inference(
    model,
    prediction_loader,
    prediction_ds,
    ckpt_path=params.CKPT_PATH,
    device=device,        
    lookahead_days=params.LOOKAHEAD_DAYS,       
    csv_path=params.CSV_PATH,
)
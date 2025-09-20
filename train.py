import os, torch
from torch.utils.data import DataLoader, random_split
from models.model import WTFailureClassifier
from datasets.data_loader import WindTurbineFailureDatasetMultiCode
from utils.train_inference_utils import balance_dataset, train, test
from utils.tools import compute_scaling_parameters
import params 

def main():
    
    # #Uncomment to compute and save new scaling parameters

    # scaler = compute_scaling_parameters(
    #     data_path="datasets/multicode_all", 
    #     save_path="scalers/wt_failure_scaler_vfinal.joblib"
    # )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    workers = min(8, os.cpu_count() or 1)

    os.makedirs(params.CKPT_DIR, exist_ok=True)

    train_ds = WindTurbineFailureDatasetMultiCode(
        root_path=params.ROOT_PATH,
        window_size=params.SEQ_LEN,
        freq=params.FREQ,
        stride=12,
        scaler_path=params.SCALER_PATH
    )

    val_test_ds = WindTurbineFailureDatasetMultiCode(
        root_path=params.ROOT_PATH,
        window_size=params.SEQ_LEN,
        freq=params.FREQ,
        stride=12,
        scaler_path=params.SCALER_PATH
    )

    print(f'Train samples: {len(train_ds):,}   |  Val + Test samples: {len(val_test_ds):,}')

    # Balanced dataset
    balanced_dataset = balance_dataset(
        dataset=train_ds,
        undersample_ratio=0.2,
        label_index=3,
        random_seed=None
    )
    train_ds=balanced_dataset

    val_len = int(0.5 * len(val_test_ds))
    test_len = len(val_test_ds) - val_len
    val_ds, test_ds = random_split(val_test_ds, [val_len, test_len])

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
        train_loader, val_loader,
        device=device,
        num_epochs=params.NUM_EPOCHS,
        patience=params.PATIENCE,
        learning_rate=params.LEARNING_RATE,
        weight_decay=params.WEIGHT_DECAY,
        ckpt_dir=params.CKPT_DIR
    )

    print(f"\n----- Testing -----")
    acc_class_0, acc_class_1=test(model, test_loader, device=device)
    print(f"  Test   | Class 0 (No‑fail) accuracy: {acc_class_0:5.2f}%")
    print(f"  Test   | Class 1 (Fail)     accuracy: {acc_class_1:5.2f}%")


if __name__ == "__main__":
    main()
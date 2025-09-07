import numpy as np
import torch
from torch import nn, optim
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm 
from torch.utils.data import Subset
import pandas as pd
from collections import Counter
from typing import Optional

class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = np.inf
        self.delta: float = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:

    model.train()
    running_loss, correct, seen = 0.0, 0, 0

    bar = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)
    for seq_x, seq_x_mark, seq_x_code, labels in bar:
        
        seq_x, seq_x_mark, seq_x_code, labels = (
            seq_x.to(device),
            seq_x_mark.to(device),
            seq_x_code.to(device),
            labels.to(device),
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(seq_x, seq_x_code, seq_x_mark)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)

        bar.set_postfix(loss=running_loss / seen, acc=f"{100.0 * correct / seen:5.2f}%")

    return running_loss / seen, 100.0 * correct / seen


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:

    model.eval()
    v_loss, v_correct, v_seen = 0.0, 0, 0
    y_true: List[int] = []
    y_pred: List[int] = []

    bar = tqdm(dataloader, total=len(dataloader), desc="Valid", leave=False)
    with torch.no_grad():
        for seq_x, seq_x_mark, seq_x_code, labels in bar:
            seq_x, seq_x_mark, seq_x_code, labels = (
                seq_x.to(device),
                seq_x_mark.to(device),
                seq_x_code.to(device),
                labels.to(device),
            )
            logits = model(seq_x, seq_x_code, seq_x_mark)
            loss = criterion(logits, labels)

            v_loss += loss.item()
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            v_correct += (preds == labels).sum().item()
            v_seen += labels.size(0)
            bar.set_postfix(loss=v_loss / v_seen, acc=f"{100.0 * v_correct / v_seen:5.2f}%")

    val_loss = v_loss / v_seen
    val_acc = 100.0 * v_correct / v_seen
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    return val_loss, val_acc, f1, prec, rec


def test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for seq_x, seq_x_mark, seq_x_code, labels in tqdm(dataloader, desc="Test", leave=False):
            seq_x, seq_x_mark, seq_x_code = (
                seq_x.to(device),
                seq_x_mark.to(device),
                seq_x_code.to(device),
            )
            logits = model(seq_x, seq_x_code, seq_x_mark)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true_np, y_pred_np = np.array(y_true), np.array(y_pred)
    acc_class_0 = compute_class_accuracy(y_true_np, y_pred_np, 0)
    acc_class_1 = compute_class_accuracy(y_true_np, y_pred_np, 1)

    print(f"  Test   | Class 0 (No‑fail) accuracy: {acc_class_0:5.2f}%")
    print(f"  Test   | Class 1 (Fail)     accuracy: {acc_class_1:5.2f}%")
    return acc_class_0, acc_class_1


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    ckpt_dir: str,
    class_weights: torch.Tensor | None = None,
) -> Dict[str, List[float]]:

    if class_weights is None:
        class_weights = torch.tensor([1.0, 10.0], device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "f1": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        #  Train 
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        #  Validate
        val_loss, val_acc, f1, prec, rec = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        #  Info 
        print(
            f"train loss {train_loss:.4f}, acc {train_acc:5.2f}% │ "
            f"val loss {val_loss:.4f}, acc {val_acc:5.2f}% │ "
            f"F1 {f1:.4f}, P {prec:.4f}, R {rec:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["f1"].append(f1)

        # Test after each epoch
        print(f"\n----- Testing after epoch {epoch + 1} -----")
        test(model, test_loader, device)

        # Persist model
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{ckpt_dir}/model_epoch_{epoch + 1}.pth")
        print(f"✅ Model saved at epoch {epoch + 1}")

        # Early stopping 
        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print(f"Early‑stopping after {epoch + 1} epoch(s)")
            break

        torch.cuda.empty_cache()

    return history


def run_inference(
    model: torch.nn.Module,
    prediction_loader,
    prediction_ds,
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    lookahead_days: int = 10,
    csv_path: str | None = "predicted_failures.csv",
    show_summary: bool = True,
) -> list[dict]:
   
    ckpt_path = Path(ckpt_path)
    csv_path = None if csv_path is None else Path(csv_path)

    #  load checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    lookahead = pd.Timedelta(days=lookahead_days)

    predicted_failure_windows: list[dict] = []
    sample_idx = 0

    # inference loop
    with torch.no_grad(), tqdm(total=len(prediction_loader.dataset),
                               desc="Inference",
                               unit="window") as pbar:

        for seq_x, seq_x_mark, seq_x_code in prediction_loader:
            batch_size     = seq_x.size(0)
            batch_indices  = range(sample_idx, sample_idx + batch_size)

            seq_x, seq_x_mark, seq_x_code = (
                seq_x.to(device),
                seq_x_mark.to(device),
                seq_x_code.to(device),
            )

            preds = model(seq_x, seq_x_code, seq_x_mark).argmax(dim=1).cpu().numpy()                             

            # Collect positive predictions
            for i, pred in enumerate(preds):
                if pred == 1:
                    ds_idx   = batch_indices[i]
                    info     = prediction_ds.get_window_info(ds_idx)
                    horizon  = info["end_date"] + lookahead

                    predicted_failure_windows.append(
                        dict(
                            sample_idx   = ds_idx,
                            start_date   = info["start_date"],
                            end_date     = info["end_date"],
                            horizon_date = horizon,
                            turbine_id   = info["turbine_id"],
                        )
                    )

            sample_idx += batch_size
            pbar.update(batch_size)

    # reporting 
    if not predicted_failure_windows:
        if show_summary:
            print("\nNo failures predicted.")
        return predicted_failure_windows

    # Sort for nicer printing
    predicted_failure_windows.sort(
        key=lambda x: (x["turbine_id"], x["horizon_date"])
    )

    if csv_path is not None:
        pd.DataFrame(predicted_failure_windows).to_csv(csv_path, index=False)
        if show_summary:
            print(f"\nPredictions saved to '{csv_path}'")

    if show_summary:
        print_inference_summary(predicted_failure_windows,
                                 total_windows=len(prediction_ds))
        
    return predicted_failure_windows

def print_inference_summary(predicted_failure_windows, *, total_windows: int):

    print("\n==== Predicted Failure Periods ====")
    current_turbine = None
    for w in predicted_failure_windows:
        if w["turbine_id"] != current_turbine:
            current_turbine = w["turbine_id"]
            print(f"\n{current_turbine}:")
        print(
            f"  Window {w['sample_idx']}: "
            f"Data {w['start_date']:%Y-%m-%d}–{w['end_date']:%Y-%m-%d}, "
            f"predicted failure ~ {w['horizon_date']:%Y-%m-%d}"
        )

    counts = Counter(w["turbine_id"] for w in predicted_failure_windows)
    print(f"\nTotal predicted failures: {len(predicted_failure_windows)}")
    print("\nPredicted failures by turbine:")
    for tu, cnt in sorted(counts.items()):
        print(f"  {tu}: {cnt}")

    all_fail_dates = [w["horizon_date"] for w in predicted_failure_windows]
    print(
        f"\nPrediction horizon range: "
        f"{min(all_fail_dates):%Y-%m-%d} – {max(all_fail_dates):%Y-%m-%d}"
    )

    normals = total_windows - len(predicted_failure_windows)
    print("\n==== Inference Summary ====")
    print(f"Total windows processed : {total_windows}")
    print(f"Predicted failures      : {len(predicted_failure_windows)} "
          f"({len(predicted_failure_windows)/total_windows*100:.1f}%)")
    print(f"Predicted normal        : {normals} "
          f"({normals/total_windows*100:.1f}%)")

def balance_dataset(dataset, undersample_ratio=0.5, label_index=3, random_seed=None):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    all_indices = list(range(len(dataset)))
    
    # Separate indices by class
    class_0_indices = [i for i in all_indices if dataset[i][label_index] == 0]
    class_1_indices = [i for i in all_indices if dataset[i][label_index] == 1]
    
    n_samples = int(len(class_0_indices) * undersample_ratio)
    class_0_sampled = np.random.choice(class_0_indices, size=n_samples, replace=False)
    balanced_indices = list(class_0_sampled) + class_1_indices
    
    # Print class distribution
    final_class_0 = sum(1 for i in balanced_indices if dataset[i][label_index] == 0)
    final_class_1 = sum(1 for i in balanced_indices if dataset[i][label_index] == 1)
    
    print(f"Original class distribution: {len(class_0_indices)} class 0, {len(class_1_indices)} class 1")
    print(f"After balancing: {final_class_0} class 0, {final_class_1} class 1")
    
    return Subset(dataset, balanced_indices)

def compute_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, class_idx: int) -> float:

    mask = y_true == class_idx
    correct = np.sum((y_pred == class_idx) & mask)
    total = np.sum(mask)
    return 100.0 * correct / total if total > 0 else 0.0

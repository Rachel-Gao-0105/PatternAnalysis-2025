import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from modules import ConvNeXt
from dataset import build_datasets
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np


def mixup_data(x, y, alpha=0.2):
    """
    Apply MixUp data augmentation.
    """
    if alpha <= 0:
        lam = 1.0
        return x, y, y, lam

    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the MixUp loss as a convex combination of two CE losses."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None, total_epochs=None):
    """
    One epoch of training with AMP + MixUp.
    Note: accuracy/AUC during training are approximate under MixUp (labels are mixed).
    """
    model.train()
    losses, prob_pos_list, hard_preds, gts = [], [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        # Dynamically decay MixUp alpha: stronger early, weaker later
        if epoch is not None and total_epochs is not None:
            current_alpha = 0.2 * max(0.0, 1.0 - epoch / (0.7 * total_epochs))
        else:
            current_alpha = 0.2

        # Apply MixUp
        x, y_a, y_b, lam = mixup_data(x, y, alpha=current_alpha)

        # AMP forward + loss
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        probs = out.softmax(1)[:, 1].detach().cpu().tolist()  
        prob_pos_list.extend(probs)
        hard_preds.extend(out.argmax(1).detach().cpu().tolist())
        gts.extend(y.detach().cpu().tolist())

    acc = accuracy_score(gts, hard_preds)
    try:
        auc = roc_auc_score(gts, prob_pos_list)
    except Exception:
        auc = float("nan")

    return sum(losses) / len(losses), acc, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Validation with AMP. Returns loss, acc (argmax), and AUC (using positive-class probability).
    """
    model.eval()
    losses, prob_pos_list, hard_preds, gts = [], [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        losses.append(loss.item())
        probs = out.softmax(1)[:, 1].cpu().tolist()
        prob_pos_list.extend(probs)
        hard_preds.extend(out.argmax(1).cpu().tolist())
        gts.extend(y.cpu().tolist())

    acc = accuracy_score(gts, hard_preds)
    try:
        auc = roc_auc_score(gts, prob_pos_list)
    except Exception:
        auc = float("nan")

    return sum(losses) / len(losses), acc, auc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to dataset root (e.g., D:/.../ADNI/AD_NC)")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.07)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--num_workers", type=int, default=0, help="Use 0 on Windows")
    ap.add_argument("--samples_per_class_train", type=int, default=None,
                    help="e.g., 100 for quick pilot; None to use full training set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10, help="Stop if no val_acc improvement for N epochs")
    ap.add_argument("--target_acc", type=float, default=0.8, help="Stop immediately if val_acc >= target_acc")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Dataset normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    #dataloaders
    train_ds, test_ds = build_datasets(
        root_dir=args.data_root,
        img_size=args.img_size,
        samples_per_class=args.samples_per_class_train,
        seed=args.seed,
        mean=mean,
        std=std
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = ConvNeXt(num_classes=args.num_classes,
                     drop_path_rate=args.drop_path_rate,
                     dropout_rate=args.dropout_rate).to(device)
    model = model.to(memory_format=torch.channels_last)

    print(">> Training with AMP + MixUp + Early Stopping + AUC")

    #optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = max(1, int(0.05 * args.epochs))   
    t_max = max(1, args.epochs - warmup_steps)      
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc = 0.0
    best_ckpt_path = os.path.join(args.out_dir, "best_acc.pth")
    no_improve_epochs = 0
    patience = args.patience

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=args.epochs
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.3f}")

        # Save best by validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "mean": mean,
                "std": std
            }, best_ckpt_path)
        else:
            no_improve_epochs += 1

        # Early stopping
        if val_acc >= args.target_acc:
            print(f"\nEarly stopping: val_acc reached {val_acc:.3f} ≥ target {args.target_acc:.3f}")
            break
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping: no val_acc improvement for {patience} epochs.")
            break

        scheduler.step()

    print(f"Best val acc: {best_acc:.3f} | best_acc_ckpt: {best_ckpt_path}")


if __name__ == "__main__":
    main()

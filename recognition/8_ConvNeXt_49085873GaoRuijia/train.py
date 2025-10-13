import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
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
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None, total_epochs=None):
    """
    Perform one epoch of training with AMP and dynamic MixUp.
    """
    model.train()
    losses, preds, gts = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)

        # Dynamically reduce MixUp strength over time
        if epoch is not None and total_epochs is not None:
            current_alpha = 0.2 * max(0.0, 1.0 - epoch / (0.7 * total_epochs))
        else:
            current_alpha = 0.2

        # Apply MixUp
        x, y_a, y_b, lam = mixup_data(x, y, alpha=current_alpha)

        # Mixed precision training (AMP)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds.extend(out.softmax(1).argmax(dim=1).cpu().tolist())
        gts.extend(y.cpu().tolist())

    acc = accuracy_score(gts, preds)
    return sum(losses)/len(losses), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, gts = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        losses.append(loss.item())
        preds.extend(out.softmax(1).argmax(dim=1).cpu().tolist())
        gts.extend(y.cpu().tolist())

    acc = accuracy_score(gts, preds)
    return sum(losses)/len(losses), acc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Path to dataset root directory")
    ap.add_argument("--img_size", type=int, default=448, help="Input image size")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.07)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--num_workers", type=int, default=0, help="Set to 0 for Windows")
    ap.add_argument("--samples_per_class_train", type=int, default=None, help="Subset size per class for quick testing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10, help="Number of epochs without improvement before stopping")
    ap.add_argument("--target_acc", type=float, default=0.8, help="Stop training early if validation acc >= target")
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
                     dropout_rate=args.dropout_rate)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    print(">> Training with AMP + MixUp + Early Stopping")

    #optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = max(1, int(0.05 * args.epochs))
    t_max = max(1, args.epochs - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    #AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc = 0.0
    best_acc_path = os.path.join(args.out_dir, "best_acc.pth")
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                          epoch=epoch, total_epochs=args.epochs)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std}, best_acc_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        scheduler.step()

        # Early stopping
        if val_acc >= args.target_acc:
            print(f"\nEarly Stopping: Validation accuracy reached {val_acc:.3f} ≥ {args.target_acc:.3f}")
            break

        if no_improve_epochs >= args.patience:
            print(f"\nEarly Stopping: No improvement for {args.patience} epochs.")
            break

    print(f"Best validation accuracy: {best_acc:.3f} | checkpoint: {best_acc_path}")


if __name__ == "__main__":
    main()

import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from modules import ConvNeXt
from dataset import build_datasets
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
from sklearn.metrics import roc_curve

def mixup_data(x, y, alpha=0.2):
    # Apply MixUp augmentation
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
    # Compute mixed loss
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=None, total_epochs=None):
    # Training for one epoch with AMP + MixUp
    model.train()
    losses, preds, gts = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        # Dynamic MixUp alpha (stronger early, weaker later)
        if epoch is not None and total_epochs is not None:
            current_alpha = 0.2 * max(0.0, 1.0 - epoch / (0.7 * total_epochs))
        else:
            current_alpha = 0.2
        
        # Apply MixUp
        x, y_a, y_b, lam = mixup_data(x, y, alpha=current_alpha)

        # AMP forward and loss
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        p = out.softmax(1)[:, 1].detach().cpu().tolist()
        preds.extend(p)
        gts.extend(y.cpu().tolist())

    acc = accuracy_score(gts, [p >= 0.5 for p in preds])
    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        auc = float("nan")
    return sum(losses)/len(losses), acc, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    # Validation loop with threshold tuning
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
        preds.extend(out.softmax(1)[:, 1].cpu().tolist())
        gts.extend(y.cpu().tolist())

    acc = accuracy_score(gts, [p >= 0.5 for p in preds])
    # AUC + best threshold tuning (Youden’s J)
    try:
        auc = roc_auc_score(gts, preds)
        fpr, tpr, thr = roc_curve(gts, preds)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_thr = float(thr[best_idx])
        acc_tuned = accuracy_score(gts, (np.array(preds) >= best_thr).astype(int))
    except Exception:
        auc = float("nan")
        best_thr = 0.5
        acc_tuned = float("nan")

    return sum(losses)/len(losses), acc, auc, acc_tuned, best_thr


def parse_args():
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g. D:/.../ADNI/AD_NC")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.07)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--samples_per_class_train", type=int, default=None, help="e.g. 100 for quick pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10, help="epochs to wait without val_acc improvement")
    ap.add_argument("--target_acc", type=float, default=0.8, help="early stop if val_acc >= target_acc")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset mean and std
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # Build datasets and dataloaders
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

    # Build model
    model = ConvNeXt(num_classes=args.num_classes, drop_path_rate=args.drop_path_rate, dropout_rate=args.dropout_rate)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)

    print(">> Training")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler: warmup + cosine annealing
    warmup_steps = max(1, int(0.05 * args.epochs))
    t_max = max(1, args.epochs - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # AMP scaler (enabled only on GPU)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Training history and metrics
    history = {"tr_loss":[], "tr_acc":[], "tr_auc":[], "val_loss":[], "val_acc":[], "val_auc":[], "val_acc_tuned":[], "val_thr":[]}
    best_acc, best_auc, best_acc_tuned = 0.0, 0.0, 0.0

    # Paths for best models
    best_acc_path = os.path.join(args.out_dir, "best_acc.pth")
    best_auc_path = os.path.join(args.out_dir, "best_auc.pth")
    best_acc_tuned_path = os.path.join(args.out_dir, "best_acc_tuned.pth")

    patience = args.patience
    no_improve_epochs = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=args.epochs
        )
        val_loss, val_acc, val_auc, val_acc_tuned, best_thr = evaluate(model, val_loader, criterion, device)

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["tr_auc"].append(tr_auc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_acc_tuned"].append(val_acc_tuned)
        history["val_thr"].append(best_thr)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} auc {tr_auc:.3f} | "
              f"val loss {val_loss:.4f} acc@0.5 {val_acc:.3f} auc {val_auc:.3f} | "
              f"val acc@bestThr {val_acc_tuned:.3f} thr={best_thr:.3f}")

        # Save best models
        if val_acc_tuned > best_acc_tuned:
            best_acc_tuned = val_acc_tuned
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_acc_tuned_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_acc_path)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "best_thr": best_thr}, best_auc_path)
                
        scheduler.step()

        # Early stopping
        if val_acc >= args.target_acc:
            print(f"\nEarly Stopping Triggered: Validation accuracy reached {val_acc:.3f} ≥ {args.target_acc:.3f}.")
            break

        if no_improve_epochs >= patience:
            print(f"\nEarly stopping due to no improvement for {patience} epochs.")
            break

    # Plot and save metrics
    plt.figure(); 
    plt.plot(history["tr_loss"]); plt.plot(history["val_loss"]); 
    plt.legend(["train","val"]); plt.title("Loss"); 
    plt.savefig(os.path.join(args.out_dir,"loss.png"))

    plt.figure()
    plt.plot(history["tr_acc"]); plt.plot(history["val_acc"]);
    plt.legend(["train","val"]); plt.title("Accuracy");
    plt.savefig(os.path.join(args.out_dir,"acc.png"))

    plt.figure()
    plt.plot(history["tr_auc"]); plt.plot(history["val_auc"]);
    plt.legend(["train","val"]); plt.title("AUC");
    plt.savefig(os.path.join(args.out_dir,"auc.png"))

    print(f"Best val acc@0.5: {best_acc:.3f} | best_acc_ckpt: {best_acc_path}")
    print(f"Best val AUC: {best_auc:.3f} | best_auc_ckpt: {best_auc_path}")
    print(f"Best val acc@bestThr: {best_acc_tuned:.3f}  | best_acc_tuned_ckpt: {best_acc_tuned_path}")

if __name__ == "__main__":
    main()

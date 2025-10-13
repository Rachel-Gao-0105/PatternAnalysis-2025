import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from modules import ConvNeXt
from dataset import build_datasets
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    losses, preds, gts = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #AMP
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().tolist())
        gts.extend(y.detach().cpu().tolist())

    acc = accuracy_score(gts, preds)
    return sum(losses)/len(losses), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, gts = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)

        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().tolist())
        gts.extend(y.detach().cpu().tolist())

    acc = accuracy_score(gts, preds)
    return sum(losses)/len(losses), acc


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g. D:/.../ADNI/AD_NC")
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--drop_path_rate", type=float, default=0.1)
    ap.add_argument("--dropout_rate", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--samples_per_class_train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #mean/std
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    #Dataloader
    train_ds, val_ds = build_datasets(
        root_dir=args.data_root,
        img_size=args.img_size,
        samples_per_class=args.samples_per_class_train,
        seed=args.seed,
        mean=mean,
        std=std
    )
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    model = ConvNeXt(num_classes=args.num_classes,
                     drop_path_rate=args.drop_path_rate,
                     dropout_rate=args.dropout_rate).to(device)

    print(">> Training with AMP enabled")

    #optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.07)

    warmup_steps = max(1, int(0.05 * args.epochs))
    t_max = max(1, args.epochs - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    #AMP Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_acc = 0.0
    best_ckpt_path = os.path.join(args.out_dir, "best_acc.pth")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "mean": mean,
                "std": std
            }, best_ckpt_path)

        scheduler.step()

    print(f"Best val acc: {best_acc:.3f} | best_acc_ckpt: {best_ckpt_path}")


if __name__ == "__main__":
    main()
